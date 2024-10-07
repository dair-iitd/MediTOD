import re
import os
from collections import Counter
import json
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from transformers import Trainer
import tempfile, subprocess

try:
    import wandb
except:
    wandb = None

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

from utils import parse_nlu, parse_pol


def post_punc(text):
    if ' _ ' in text:
        text = text.replace(' _ ', '_')
    if '@ ' in text:
        text = text.replace('@ ', '@')
    if ' @' in text:
        text = text.replace(' @', '@')
    if ' = ' in text:
        text = text.replace(' = ', '=')

    return text


class OsceTrainer(Trainer):
    def __init__(self, cfg, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.cfg = cfg

    def decode_responses(self, inputs, outputs):
        tokenizer = self.tokenizer
        eos = self.tokenizer.eos_token

        if 't5' not in self.cfg['model']['wildcard']:
            input_end = inputs.size(1)
            outputs = outputs[:, input_end:]
        preds = []
        responses = tokenizer.batch_decode(
            outputs, clean_up_tokenization_spaces=False,
            skip_special_tokens=True
        )
        for resp in responses:
            pred = resp.split(eos, 1)[0]
            pred = pred.strip()
            if 'biogpt' in self.cfg['model']['wildcard']:
                pred = post_punc(pred)
            preds.append(pred.strip())

        return preds

    def run_evaluation(self, dataset):
        local_rank = self.args.local_rank
        world_size = max(self.args.world_size, 1)

        max_new_tokens = self.cfg['dev']['max_resp_length']
        # pred_end = self.vocab.eos_token_idx
        model = self._wrap_model(self.model, training=False)
        model.eval()

        if type(model) == DataParallel:
            model = model.module

        if world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            sampler=sampler,
        )
        model.eval()
        if world_size > 1:
            sampler.set_epoch(0)
        loader = tqdm(dataloader, desc='Evaluation...') if local_rank in [-1, 0] else dataloader
        responses = []
        for inputs in loader:
            batch = dict([(k, v.to(self.model.device)) for k, v in inputs.items()])
            if self.cfg['dev'].get('sample', False):
                with torch.no_grad():
                    outputs = model.generate(
                        **batch, use_cache=True,
                        max_new_tokens=max_new_tokens,
                        do_sample=True, min_length=1,
                        temperature=self.cfg['dev'].get('temperature', 0.85),
                        top_k=self.cfg['dev'].get('top_k', 8),
                        top_p=self.cfg['dev'].get('top_p', 0.9),
                    )
            else:
                with torch.no_grad():
                    outputs = model.generate(
                        **batch, use_cache=True,
                        max_length=self.cfg['model']['model_max_length'],
                        do_sample=False,
                        num_beams=self.cfg['dev'].get('num_beams', 1),
                    )

            outputs = outputs.to('cpu')
            responses.extend(self.decode_responses(inputs['input_ids'], outputs))

        model.train()
        if world_size > 1:
            all_responses = [None for _ in range(self.args.world_size)]
            dist.all_gather_object(all_responses, responses)
        else:
            all_responses = [responses]

        final_responses = []
        for ii in range(len(responses)):
            for resps in all_responses:
                final_responses.append(resps[ii])
        final_responses = final_responses[:len(dataset)]

        return final_responses

    def get_metrics(self, responses, dataset):
        raise NotImplementedError

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
        save_results=False,
        result_path=None,
    ):
        print(f'Running evaluation......{self.args.local_rank}')
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        responses = self.run_evaluation(eval_dataset)
        tmetrics = self.get_metrics(responses, eval_dataset)
        print(responses[:10])

        metrics = dict()
        for k, v in tmetrics.items():
            metrics[f"{metric_key_prefix}_{k}"] = v

        if wandb is not None and wandb.run is not None:
            wandb.log(metrics)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        print(f'Evaluation results: {json.dumps(metrics, indent=2)}')

        if save_results and self.args.local_rank in [-1, 0]:
            ret = []
            for ii, resp in enumerate(responses):
                ret.append(dict())
                ret[-1]['dial_id'] = eval_dataset[ii]['dial_id']
                ret[-1]['turn_num'] = eval_dataset[ii]['turn_num']
                ret[-1]['prediction'] = resp
                ret[-1]['input'] = eval_dataset[ii]['input_seq']
                ret[-1]['output'] = eval_dataset[ii]['output']

            tname = f'results_{self.state.global_step}.json'
            if result_path is not None:
                tname = result_path
            else:
                tname = os.path.join(
                    self.args.output_dir, f'results_{self.state.global_step}.json'
                )
            with open(tname, 'w') as fp:
                json.dump(ret, fp, indent=2)

        return metrics


def get_nlu_key_value_pairs(nlu):
    value_key = [
        'negative_exposure',
        'negative_family_history',
        'negative_habit',
        'negative_medical_history',
        'negative_medication',
        'negative_symptom',

        'positive_exposure',
        'positive_family_history',
        'positive_habit',
        'positive_medical_history',
        'positive_medication',
        'positive_symptom',

        'unknown_exposure',
        'unknown_family_history',
        'unknown_habit',
        'unknown_medical_history',
        'unknown_symptom',

        'avail_medical_test',
        'unavail_medical_test',
        'occupation',
        'residence',

        # Error but okay
        'basic_information',
    ]

    kvs = []
    for entry in nlu:
        intent = entry['intent']
        slots = entry.get('slots', [])

        if len(slots) == 0:
            kvs.append((intent,))
            continue

        for slot, data in slots.items():
            if slot in value_key:
                KEY = 'value'
            elif slot == 'travel':
                KEY = 'destination'
            else:
                print('Error....', slot)
                KEY = slot

            assert type(data) == list
            for ee in data:
                assert type(ee) == dict
                prm = ee.get(KEY, 'dummy')
                for kk, vv in ee.items():
                    if type(vv) == list:
                        kvs.extend([(intent, slot, prm, kk, tvv) for tvv in vv])
                    else:
                        kvs.append((intent, slot, prm, kk, vv))
    return set(kvs)


def bspan_to_constraint_dict(bspan):
    list_fields = {
        'aggravating_factor',
        'alleviating_factor',
        'location',
        'negative_characteristics',
        'not_aggravating_factor',
        'not_alleviating_aggravating_factor',
        'not_alleviating_factor',
        'positive_characteristics',
        'unknown_ana_factor',
        'unknown_characteristics',
        'unknown_factor'
    }

    text = bspan.replace('<sos_b>', '')
    text = text.replace('<eos_b>', '')
    text = text.strip()

    # pattern = r"\[[a-z_]+\]\s*\[[a-z_]+\]"
    pattern = r"@[\-a-z_]+@"
    entry_sidxs = []
    for ee in re.finditer(pattern, text):
        span = ee.span()
        assert span[1] - span[0] > 0
        entry_sidxs.append(span[0])
    entry_sidxs.append(len(text))

    intent_data = []
    for ii in range(len(entry_sidxs) - 1):
        ttext = deepcopy(text[entry_sidxs[ii]:entry_sidxs[ii + 1]])
        _, intent, ttext = ttext.split('@', 2)
        intent = intent.strip()

        ttext = ttext.strip()
        pattern = r"\[[a-z_]+\]"
        sidxs = []
        for ee in re.finditer(pattern, ttext):
            span = ee.span()
            assert span[1] - span[0] > 0
            sidxs.append(span[0])
        sidxs.append(len(ttext))

        slot_data = dict()
        for jj in range(len(sidxs) - 1):
            vspan = ttext[sidxs[jj]:sidxs[jj + 1]]
            key, tmp = vspan.split(']', 1)
            key = key.strip('[').strip()
            tmp = tmp.strip()

            pattern = r"\(.*?\)"
            kvs_arr = []
            for ee in re.finditer(pattern, tmp):
                ss, ee = ee.span()
                vts = tmp[ss:ee].strip('(').strip(')').strip()
                kvs = dict()
                for vt in vts.split('#'):
                    kk, vv = vt.split('=')
                    kk = kk.strip()
                    vv = vv.strip()
                    if kk in list_fields:
                        vv = vv.strip().split('|')
                        vv = [z.strip() for z in vv]
                    kvs[kk.strip()] = vv
                kvs_arr.append(kvs)

            slot_data[key] = kvs_arr

        intent_data.append({
            'intent': intent,
            'slots': slot_data
        })

    return intent_data


class NluTrainer(OsceTrainer):
    def __init__(self, cfg, tokenizer, **kwargs):
        super().__init__(cfg, tokenizer, **kwargs)

    def get_metrics(self, responses, dataset):
        tp, fp, fn = 0, 0, 0
        corr = 0
        parsing_errors = 0

        for ii, resp in enumerate(responses):
            try:
                pred = bspan_to_constraint_dict(resp)
            except:
                pred = []
                parsing_errors += 1

            pred = get_nlu_key_value_pairs(pred)
            gold = bspan_to_constraint_dict(dataset[ii]['bspn_nlu'])
            gold = get_nlu_key_value_pairs(gold)

            corr += int(pred == gold)
            tp += len(gold.intersection(pred))
            fp += len(pred - gold)
            fn += len(gold - pred)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        acc = corr / len(responses)
        parsing_errors = parsing_errors / len(responses)

        return {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": acc,
            "parsing_errors": parsing_errors
        }


def get_pol_key_value_pairs(actions):
    canon_slots = [
        'exposure',
        'family_history',
        'habit',
        'medical_history',
        'medication',
        'symptom',
        'medical_test',
        # Diagnosis
        'disease'
    ]

    all_kvs = set()
    for adata in actions:
        loc_kvs = set()
        act = adata['action']
        for kk in adata:
            if kk == 'action':
                continue

            for vv in adata[kk]:
                # assert all(x in ['value', 'checks'] for x in vv)
                head = vv.get('value', 'dummy')
                if 'value' in vv:
                    loc_kvs.add((act, kk, head, 'value', vv['value']))
                for xx in vv.get('checks', []):
                    ctype = xx.get('type', 'dummy')
                    if 'type' in xx:
                        loc_kvs.add((act, kk, head, 'checks', ctype))
                    for yy in xx.get('values', []):
                        loc_kvs.add((act, kk, head, f'{ctype}-value', yy))
                for gg in vv:
                    if gg in ['value', 'checks']:
                        continue
                    if type(vv[gg]) == list:
                        for zz in vv[gg]:
                            loc_kvs.add((act, kk, head, gg, zz))
                    else:
                        loc_kvs.add((act, kk, head, gg, vv[gg]))

            if len(loc_kvs) == 0:
                loc_kvs.add((act,))
            all_kvs = all_kvs.union(loc_kvs)

    return set(all_kvs)


def aspan_to_constraint_dict(aspan):
    text = aspan.replace('<sos_a>', '')
    text = text.replace('<eos_a>', '')
    text = text.strip()

    pattern = r"@[\-a-z_]+@"
    entry_sidxs = []
    for ee in re.finditer(pattern, text):
        span = ee.span()
        assert span[1] - span[0] > 0
        entry_sidxs.append(span[0])
    entry_sidxs.append(len(text))

    ret_data = []
    for ii in range(len(entry_sidxs) - 1):
        ttext = deepcopy(text[entry_sidxs[ii]:entry_sidxs[ii + 1]])
        _, action, ttext = ttext.split('@', 2)
        action = action.strip()

        ttext = ttext.strip()
        pattern = r"\[[a-z_]+\]"
        sidxs = []
        for ee in re.finditer(pattern, ttext):
            span = ee.span()
            assert span[1] - span[0] > 0
            sidxs.append(span[0])
        sidxs.append(len(ttext))

        action_data = {
            'action': action
        }
        for jj in range(len(sidxs) - 1):
            vspan = ttext[sidxs[jj]:sidxs[jj + 1]]
            key, tmp = vspan.split(']', 1)
            key = key.strip('[').strip()
            tmp = tmp.strip()

            pattern = r"\(.*?\)"
            act_data = []
            for ee in re.finditer(pattern, tmp):
                ss, ee = ee.span()
                vts = tmp[ss:ee].strip('(').strip(')').strip()
                checks = dict()
                kvs = dict()
                for vt in vts.split('#'):
                    kk, vv = vt.split('=')
                    kk = kk.strip()
                    vv = vv.strip()

                    if 'values' in kk:
                        vv = vv.strip().split('|')
                        vv = [z.strip() for z in vv]

                    # kk and vv are here.
                    if 'checks-type' == kk:
                        checks[vv] = {
                            'type': vv
                        }
                    elif '-values' in kk:
                        ctype = kk.split('-', 1)[0].strip()
                        checks[ctype]['values'] = vv
                    else:
                        kvs[kk] = vv
                if len(checks) > 0:
                    kvs['checks'] = [checks[kk] for kk in checks]
                act_data.append(kvs)
            action_data[key] = act_data
        ret_data.append(action_data)

    return ret_data


class PolTrainer(OsceTrainer):
    def __init__(self, cfg, tokenizer, **kwargs):
        super().__init__(cfg, tokenizer, **kwargs)

    def get_metrics(self, responses, dataset):
        tp, fp, fn = 0, 0, 0
        corr = 0
        parsing_errors = 0

        for ii, resp in enumerate(responses):
            try:
                pred = aspan_to_constraint_dict(resp)
            except:
                pred = []
                parsing_errors += 1
            
            pred = get_pol_key_value_pairs(pred)
            gold = aspan_to_constraint_dict(dataset[ii]['aspn'])
            gold = get_pol_key_value_pairs(gold)

            corr += int(pred == gold)
            tp += len(gold.intersection(pred))
            fp += len(pred - gold)
            fn += len(gold - pred)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        acc = corr / len(responses)
        parsing_errors = parsing_errors / len(responses)

        return {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": acc,
            "parsing_errors": parsing_errors
        }


def moses_multi_bleu(hypotheses, references, lowercase=False):
    """Calculate the bleu score for hypotheses and references
    using the MOSES ulti-bleu.perl script.
    Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.
    lowercase: If true, pass the "-lc" flag to the multi-bleu script
    Returns:
    The BLEU score as a float32 value.
    """

    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    # Get MOSES multi-bleu script
    multi_bleu_path = os.path.abspath("./multi-bleu.perl")

    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()

     # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                print("multi-bleu.perl script returned non-zero exit code")
                print(error.output)
                bleu_score = np.float32(0.0)

    # Close temp files
    hypothesis_file.close()
    reference_file.close()
    return bleu_score


class NlgTrainer(OsceTrainer):
    def __init__(self, cfg, tokenizer, **kwargs):
        super().__init__(cfg, tokenizer, **kwargs)

    def get_metrics(self, responses, dataset):
        pred = []
        for resp in responses:
            words = resp.split()
            if words[0] == '<sos_r>':
                words = words[1:]
            elif words[-1] == '<eos_r>':
                words = words[:-1]
            pred.append(' '.join(words))

        gold = [dataset[ii]['output'] for ii in range(len(dataset))]
        bleu_res = moses_multi_bleu(pred, gold)

        return {
            'bleu': bleu_res
        }
