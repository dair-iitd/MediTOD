import re
import json
import yaml
import argparse

import os
import torch
from transformers import BitsAndBytesConfig
from datasets import Dataset

train_tag = 'train'
val_tag = 'valid'
test_tag = 'test'


def read_cli():
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument(
        "-cfg",
        "--config",
        help="Path to config file",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-model_path",
        "--model_path",
        help="Path to checkpoint",
        required=False,
        type=str,
        default=None
    )
    # OVERRIDES
    parser.add_argument(
        "-datapath",
        "--datapath",
        help="Data path to use instead of training",
        required=False,
        type=str,
        default=None
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Learning rate",
        required=False,
        type=float,
        default=None
    )
    parser.add_argument(
        "-fp16",
        "--fp16",
        help="Enable FP16",
        required=False,
        type=str,
        default=None,
        choices=['True', 'False']
    )
    parser.add_argument(
        "-warmup_ratio",
        "--warmup_ratio",
        help="Warm up ratio",
        required=False,
        type=float,
        default=None
    )
    parser.add_argument(
        "-bsz",
        "--batch_size",
        help="Batch size",
        required=False,
        type=int,
        default=None
    )
    parser.add_argument(
        "-seed",
        "--seed",
        help="Seed",
        required=False,
        type=int,
        default=None
    )
    parser.add_argument(
        "-rs",
        "--result_path",
        help="Result path",
        required=False,
        type=str,
        default=None
    )
    parser.add_argument(
        "-it",
        "--infer_tag",
        help="Infer Path",
        required=False,
        type=str,
        default=None
    )
    args = vars(parser.parse_args())

    return args


def override_config(cfg, ocfg):
    tags = []

    if ocfg['datapath'] is not None:
        cfg['datapath'] = ocfg['datapath']
        tags.append(('datapath', ocfg['datapath']))

    if ocfg['learning_rate'] is not None:
        cfg['train']['learning_rate'] = ocfg['learning_rate']
        tags.append(('learning_rate', ocfg['learning_rate']))

    if ocfg['fp16'] is not None:
        cfg['train']['fp16'] = ocfg['fp16']
        tags.append(('fp16', ocfg['fp16']))

    if ocfg['warmup_ratio'] is not None:
        cfg['train']['warmup_ratio'] = ocfg['warmup_ratio']
        tags.append(('warmup_ratio', ocfg['warmup_ratio']))

    if ocfg['batch_size'] is not None:
        cfg['train']['per_device_train_batch_size'] = ocfg['batch_size']
        tags.append(('batch_size', ocfg['batch_size']))

    if ocfg['seed'] is not None:
        cfg['train']['seed'] = ocfg['seed']
        tags.append(('seed', ocfg['seed']))

    if ocfg['model_path'] is not None:
        cfg['model_path'] = ocfg['model_path']

    if ocfg['result_path'] is not None:
        cfg['result_path'] = ocfg['result_path']

    if ocfg['infer_tag'] is not None:
        cfg['infer_tag'] = ocfg['infer_tag']

    if len(tags) > 0:
        print(tags)
        tag = '_'.join([
            f"{k}:{v}" for (k, v) in tags
        ])
        cfg['experiment_name'] = cfg['experiment_name'] + '_' + tag

    return cfg


def get_config(config_file):
    print(f'Reading config from', config_file)
    with open(config_file, 'r') as fp:
        cfg = yaml.safe_load(fp)
    
    return cfg


def get_joint_config(wandb_cfg):
    base_cfg = get_config(wandb_cfg['base_config'])
    base_cfg['base_name'] = base_cfg['experiment_name']

    ov_keys = [base_cfg['experiment_name']]
    for key1 in wandb_cfg.keys():
        if key1 not in base_cfg:
            continue
        for key2 in wandb_cfg[key1].keys():
            base_cfg[key1][key2] = wandb_cfg[key1][key2]
            ov_keys.append(f"{key1}-{key2}:{wandb_cfg[key1][key2]}")
    base_cfg['experiment_name'] = '_'.join(ov_keys)

    return base_cfg


def load_json(fname):
    with open(fname, 'r') as fp:
        obj = json.load(fp)
    return obj


def parse_nlu(response):
    # [action 1] (values) a, b, c (checks) e, f, g [action 2]..
    words = [x.strip() for x in response.split()]
    if len(words) == 0:
        return []

    if words[0] == '<sos_b>':
        words = words[1:]
    if words[-1] == '<eos_b>':
        words = words[:-1]
    if words[0] == '[medical]':
        words = words[1:]

    aidxs = [ii for ii in range(len(words)) if re.search(r"\[([a-zA-Z0-9_\-]+)\]", words[ii]) is not None]
    aidxs.append(len(words))
    actions = []
    for ii, st in enumerate(aidxs[:-1]):
        en = aidxs[ii + 1]
        act = dict()
        act['intent'] = words[st][1:-1]

        vidxs = [jj for jj in range(st + 1, en) if re.search(r"\(([a-zA-Z0-9_]+)\)", words[jj]) is not None]
        vidxs.append(en)

        slots = dict()
        for jj, vst in enumerate(vidxs[:-1]):
            ven = vidxs[jj + 1]
            if words[vst] not in slots:
                slots[words[vst][1:-1]] = []
            vvs = ' '.join(words[vst + 1:ven]).split(',')
            vvs = [x.strip() for x in vvs]
            vvs = [x for x in vvs if len(x) > 0]
            slots[words[vst][1:-1]].extend(vvs)

        if len(slots) > 0:
            act['slots'] = slots
        for kk in list(act):
            if len(act[kk]) == 0:
                del act[kk]
        actions.append(act)

    return actions, True


def parse_pol(response):
    # [action 1] (values) a, b, c (checks) e, f, g [action 2]..
    words = [x.strip() for x in response.split()]

    aidxs = [ii for ii in range(len(words)) if re.search(r"\[([a-zA-Z0-9_\-]+)\]", words[ii]) is not None]
    aidxs.append(len(words))
    actions = []
    for ii, st in enumerate(aidxs[:-1]):
        en = aidxs[ii + 1]
        act = dict()
        act['action'] = words[st][1:-1]

        vidxs = [jj for jj in range(st + 1, en) if re.search(r"\(([a-zA-Z0-9_]+)\)", words[jj]) is not None]
        vidxs.append(en)

        for jj, vst in enumerate(vidxs[:-1]):
            ven = vidxs[jj + 1]
            if words[vst] not in act:
                act[words[vst][1:-1]] = []
            vvs = ' '.join(words[vst + 1:ven]).split(',')
            vvs = [x.strip() for x in vvs]
            vvs = [x for x in vvs if len(x) > 0]
            act[words[vst][1:-1]].extend(vvs)

        for kk in list(act):
            if len(act[kk]) == 0:
                del act[kk]
        actions.append(act)

    return actions, True


def get_session_samples(session, history_size, system_message, tasks, ignore_system=False):
    data = []
    context = []

    for entry in session:
        context.append(
            f"patient: {entry['user'].strip('<sos_u>').strip('<eos_u>').strip()}"
        )

        for task in ['nlu', 'pol', 'nlg']:
            if task not in tasks:
                continue

            clen = len(context)
            st = max(clen - history_size, 0)
            en = len(context)
            prompt = ''

            if task == 'nlu':
                if en - st > 2:
                    text = '\n'.join(context[st:-2])
                    prompt = "[dialog history]\n" + text + "\n\n"
                
                text = '\n'.join(context[-2:])
                prompt += "[last turn]\n" + text + "\n\n[output]"

            elif task == 'pol':
                prompt += '[dialog state]\n' + json.dumps(entry['dst']) + '\n\n'

                if en - st > 2:
                    text = '\n'.join(context[st:-2])
                    prompt += "[dialog history]\n" + text + "\n\n"

                text = '\n'.join(context[-2:])
                prompt += "[last turn]\n" + text + "\n\n[output]"

            elif task == 'nlg':
                prompt += '[action]\n' + json.dumps(entry['pol']) + '\n\n'
                text = '\n'.join(context[-2:])
                prompt += "[last turn]\n" + text + "\n\n[output]"

            sample = dict()
            sample['dial_id'] = entry['dial_id']
            sample['turn_num'] = entry['turn_num']
            if ignore_system:
                sample['prompt_input'] = [
                    {'role': 'user', 'content': system_message[task] + '\n\n' + prompt}
                ]
            else:
                sample['prompt_input'] = [
                    {'role': 'system', 'content': system_message[task]},
                    {'role': 'user', 'content': prompt}
                ]

            ttt = json.dumps(entry[task]) if task != 'nlg' else entry[task]
            sample['prompt_output'] = [
                {'role': 'assistant', 'content': '[answer] ' + ttt + ' [done]'}
                # {'role': 'assistant', 'content': json.dumps(entry[task])}
            ]
            sample['task'] = task
            sample['task_output'] = entry[task]
            data.append(sample)

        context.append(
            f"doctor: {entry['resp'].strip('<sos_r>').strip('<eos_r>').strip()}"
        )

    return data


def load_data(datapath, tag, tasks, history_size, system_message=None, ignore_system=False):
    """Loads dataset from the datapath"""
    if system_message is None:
        system_message = {
            'nlu': "You are a professional medical scribe who is an expert in understanding doctor-patient dialogs. The user will show you a dialog history between a doctor and a patient and the last turn in their dialog. Your task is to identify the patient's intent, slots, and related attributes (if applicable) from the given the dialog history and the last turn. Definitions for intent, slots, and related attributes are given below as Python dictionaries.",
            'pol': "You are a professional medical assistant who is an expert in understanding doctor-patient dialogs. The user will show you current state, (partial) history and last turn of a dialog between a doctor and a patient. Your task is to suggest the doctor's action as a continuation of the dialog. Doctor's action consists of an action and related attributes (if applicable).",
            'nlg': "You are a professional medical assistant who is an expert in understanding doctor-patient dialogs. The user will show you the last turn of the dialog between a doctor and a patient and the doctor's action. Your task is to suggest the doctor's response as a continuation of the dialog."
        }

    ttag = tag
    if tag == 'valid':
        ttag = 'dev'
    with open(os.path.join(datapath, f"fine-processed-{ttag}.json"), "r") as fp:
        raw_data = json.load(fp)

    print(f"Loaded {len(raw_data)} dialogs for {tag}...")

    data = []
    for session in raw_data:
        samples = get_session_samples(session, history_size, system_message, tasks, ignore_system)
        data.extend(samples)

    print(f'Total samples: {len(data)}')
    dataset = Dataset.from_list(data)

    return dataset


def get_bitsandbytes_config(quantization=None):
    if torch.cuda.get_device_capability()[0] >= 8 or quantization == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False, # We don't know if this is okay.
        )

    return BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False, # We don't know if this is okay.
    )
