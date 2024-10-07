import random
import numpy as np
import json
import argparse
from tqdm import tqdm
from copy import deepcopy

np.random.seed(43)
random.seed(43)

from dynamic import DynamicPrompter, DynamicPolicyPrompter, DynamicNLGPrompter


def load_json(fname):
    with open(fname, 'r') as fp:
        obj = json.load(fp)
    return obj


def read_cli():
    parser = argparse.ArgumentParser(description='Prompt Gen')
    parser.add_argument('--raw_file', help='Path to raw dialogs file.', required=True, type=str)
    parser.add_argument('--train_file', help='Path to train index file.', required=True, type=str)
    parser.add_argument('--test_file', help='Path to test index file', required=True, type=str)
    parser.add_argument('--tar_file', help='Path to dest file', required=True, type=str)
    parser.add_argument('--history_size', help='History size', default=4, required=False, type=int)
    parser.add_argument('--nexp', help='Num Exemplar', required=False, type=int, default=5)
    parser.add_argument('--task', help='task', required=True, type=str, choices=['nlu', 'pol', 'nlg'])
    # parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose mode')
    
    args = parser.parse_args()
    args = vars(args)

    return args
    

def run(args):
    dialogs = load_json(args['raw_file'])
    train_ids = load_json(args['train_file'])
    test_ids = load_json(args['test_file'])

    assert args['task'] in ['nlu', 'pol', 'nlg']
    if args['task'] == 'nlu':
        prompter = DynamicPrompter(
            num_exemplars=args['nexp'], history_size=args['history_size'],
            dialogs=dialogs, train_ids=train_ids
        )
    elif args['task'] == 'pol':
        prompter = DynamicPolicyPrompter(
            num_exemplars=args['nexp'], history_size=args['history_size'],
            dialogs=dialogs, train_ids=train_ids
        )
    elif args['task'] == 'nlg':
        prompter = DynamicNLGPrompter(
            num_exemplars=args['nexp'], history_size=args['history_size'],
            dialogs=dialogs, train_ids=train_ids
        )

    prompts = []
    for did in tqdm(test_ids):
        for uid, uttr in enumerate(dialogs[did]['utterances']):
            if args['task'] == 'nlu':
                if uttr['speaker'] != 'patient':
                    continue

                if uid + 1 == len(dialogs[did]['utterances']):
                    continue

                st = max(0, uid - args['history_size'])
                en = uid + 1
                context = [
                    f"{dialogs[did]['utterances'][ii]['speaker']}: {dialogs[did]['utterances'][ii]['text']}"
                    for ii in range(st, en, 1)
                ]

                dialog_history = context[:-2]
                last_turn = context[-2:]

                prompt_elements = prompter.get_prompt_elements(
                    dialog_history, last_turn
                )

                prompts.append({
                    'did': did, 'uid': uid,
                    'prompt_elements': prompt_elements,
                    'nlu': deepcopy(uttr['nlu'])
                })

            elif args['task'] == 'pol':
                if uttr['speaker'] != 'doctor':
                    continue

                st = max(0, uid - args['history_size'] - 1)
                en = uid
                context = [
                    f"{dialogs[did]['utterances'][ii]['speaker']}: {dialogs[did]['utterances'][ii]['text']}"
                    for ii in range(st, en, 1)
                ]

                if uid == 0:
                    dialog_state = dict()
                else:
                    dialog_state = dialogs[did]['utterances'][uid - 1]['dialog_state']

                # dialog_history = context[:-2]
                last_turn = context[-2:]
                prompt_elements = prompter.get_prompt_elements(
                    dialog_state, last_turn
                )

                prompts.append({
                    'did': did, 'uid': uid,
                    'prompt_elements': prompt_elements,
                    'pol': deepcopy(uttr['actions'])
                })

            elif args['task'] == 'nlg':
                if uttr['speaker'] != 'doctor':
                    continue

                st = max(0, uid - args['history_size'] - 1)
                en = uid
                context = [
                    f"{dialogs[did]['utterances'][ii]['speaker']}: {dialogs[did]['utterances'][ii]['text']}"
                    for ii in range(st, en, 1)
                ]

                if uid == 0:
                    actions = dict()
                else:
                    actions = dialogs[did]['utterances'][uid]['actions']

                # dialog_history = context[:-2]
                last_turn = context[-2:]
                prompt_elements = prompter.get_prompt_elements(
                    actions, last_turn
                )

                prompts.append({
                    'did': did, 'uid': uid,
                    'prompt_elements': prompt_elements,
                    'nlg': deepcopy(dialogs[did]['utterances'][uid]['text'])
                })

    print(f'Created {len(prompts)} prompts...')
    with open(args['tar_file'], 'w') as fp:
        json.dump(prompts, fp)


if __name__ == "__main__":
    args = read_cli()
    run(args)
