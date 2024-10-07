import json
import os
from formating_utils import deformat_nlu
import argparse


def read_cli():
    parser = argparse.ArgumentParser(description='Prompt Gen')
    parser.add_argument('--path', help='Path to raw predictions.', required=True, type=str)
    parser.add_argument('--tar_file', help='Path to target file.', required=True, type=str)
    parser.add_argument('--model', help='Model', required=True, type=str, choices=['gpt-3.5-turbo', 'biollm'])
    parser.add_argument('--data', help='data', required=True, type=str)
    parser.add_argument('--task', help='Task.', required=True, type=str, choices=['nlu', 'pol', 'nlg'])
    args = parser.parse_args()
    args = vars(args)

    return args


def post_process(text, model):
    if text is None:
        return ''
    if 'gpt' in model:
        return text.lower()

    elif model == 'biollm':
        text = text.lower()
        text = text.split('<|im_end|>', 1)[0]
        text = text.strip()

        return text

def run(args):
    dnames = os.listdir(args['path'])
    dnames = sorted(dnames, key=lambda x: int(x[:-5]))
    with open(args['data'], 'r') as fp:
        dialogs = json.load(fp)

    parsing_errs = 0
    ret = []
    for dname in dnames:
        fname = os.path.join(args['path'], dname)
        with open(fname, 'r') as fp:
            obj = json.load(fp)

        if args['task'] in ['pol', 'nlg'] and obj['uid'] == 0:
            print('SKIPPED')
            continue

        prediction = post_process(obj['prediction'], args['model'])
        text = post_process(obj['prediction'], args['model'])
        if args['task'] == 'nlg':
            gg = dialogs[obj['did']]['utterances'][obj['uid']]['text']
        else:
            parsing_err = False
            try:
                prediction = json.loads(text)
                if args['task'] == 'nlu':
                    prediction = deformat_nlu(prediction)
            except Exception as e:
                print()
                print('#', dname, obj['prediction'], '#')
                print()
                parsing_err = True
                prediction = []

            parsing_errs += int(parsing_err)
            if args['task'] == 'nlu':
                gg = dialogs[obj['did']]['utterances'][obj['uid']]['nlu']
            else:
                gg = dialogs[obj['did']]['utterances'][obj['uid']]['actions']

        ret.append({
            'did': obj['did'], 'uid': obj['uid'],
            'prediction': prediction,
            # 'nlu': obj['nlu'],
            # 'pol': obj['pol'],
            'nlg': dialogs[obj['did']]['utterances'][obj['uid']]['text'],
            args['task']: gg,
        })

    print('Parsing errors', parsing_errs, parsing_errs / len(dnames))
    with open(args['tar_file'], 'w') as fp:
        json.dump(ret, fp)


if __name__ == '__main__':
    args = read_cli()
    run(args)
