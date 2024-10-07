import json
import numpy as np

import re
import argparse
from copy import deepcopy
from tqdm import tqdm

import os
import json
import numpy as np
import subprocess, tempfile
import re
import string
from nltk.translate.bleu_score import corpus_bleu




def read_cli():
    parser = argparse.ArgumentParser(description='Evaluation POL')
    parser.add_argument(
        "-data",
        "--data",
        help="Path to dialog.json",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-pred",
        "--pred",
        help="Path to predictions",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-model",
        "--model",
        help="Model",
        required=True,
        choices=['PPTOD', 'Flan-T5', 'BioGPT', 'LLM'],
        type=str,
    )
    parser.add_argument(
        "-mode",
        "--mode",
        help="mode",
        default='all',
        required=False,
        choices=['canon_only', 'all'],
        type=str,
    )
    args = vars(parser.parse_args())

    return args


def preprocess_text(text):
    """Preprocess utterance and table value."""
    text = text.strip().replace("\t", " ").lower()
    for p in string.punctuation:
        text = text.replace(p, f" {p} ")
    text = " ".join(text.split())
    return text


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


def load_json(fname):
    with open(fname, 'r') as fp:
        obj = json.load(fp)
    return obj


def compute_bleu_score(preds, golds):
    hypotheses = [
        x.split() for x in preds
    ]
    references = [
        [x.split()] for x in golds
    ]

    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu2 * 100, bleu4 * 100


def compute_bert_score(preds, golds):
    from bert_score import score

    P, R, F1 = score(preds, golds, lang='en', verbose=True)

    score = F1.mean()
    return score.item()


def compute_rouge_score(preds, golds):
    import evaluate

    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=preds,
        references=golds,
        rouge_types=['rouge1', 'rougeL'],
        use_aggregator=True
    )

    return results


def load_json(fname):
    with open(fname, 'r') as fp:
        obj = json.load(fp)
    return obj


if __name__ == '__main__':
    args = read_cli()

    data = load_json(args['data'])

    gold_pols = []
    pred_pols = []
    ordered_dids_uids = []

    if args['model'] == 'PPTOD':
        pass

    elif args['model'] in ['Flan-T5', 'BioGPT']:
        pred_data = load_json(args['pred'])

        did2samples = dict()
        for entry in pred_data:
            did = entry['dial_id'][4:]
            if did not in did2samples:
                did2samples[did] = dict()
            did2samples[did][entry['turn_num']] = entry

        dids = list(did2samples)
        for ii, did in enumerate(dids):
            loc_gold = []
            loc_pred = []

            dlg = data[did]
            uids = sorted(did2samples[did], key=lambda x: int(x))
            pids = [
                jj for jj in range(len(dlg['utterances']))
                if dlg['utterances'][jj]['speaker'] == 'doctor' and jj != 0
            ]

            for jj, uid in enumerate(uids):
                ordered_dids_uids.append((did, pids[jj]))
                gold = dlg['utterances'][pids[jj]]['text']
                loc_gold.append(gold)

                pred = did2samples[did][uid]['prediction']
                loc_pred.append(pred)

            gold_pols.append(loc_gold)
            pred_pols.append(loc_pred)


    elif args['model'] in ['LLM']:
        pred_data = load_json(args['pred'])
        curr_did = pred_data[0]['did']
        loc_gold = []
        loc_pred = []
        for entry in pred_data:
            did = entry['did']
            if did != curr_did:
                gold_pols.append(loc_gold)
                pred_pols.append(loc_pred)

                loc_gold = []
                loc_pred = []

            loc_gold.append(entry['nlg'])
            loc_pred.append(entry['prediction'])

        gold_pols.append(loc_gold)
        pred_pols.append(loc_pred)

    print(loc_gold[:5])
    print(loc_pred[:5])
    golds = [preprocess_text(x) for y in gold_pols for x in y]
    preds = [preprocess_text(x) for y in pred_pols for x in y]

    ret = dict()
    bleu2, bleu4 = compute_bleu_score(preds, golds)
    ret['BLEU-2'] = bleu2
    ret['BLEU-4'] = bleu4

    ret['bert_score'] = compute_bert_score(preds, golds)
    ret.update(compute_rouge_score(preds, golds))

    print(json.dumps(ret, indent=2))

