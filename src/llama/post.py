import json
import os
import argparse
from collections import Counter


def read_cli():
    parser = argparse.ArgumentParser(description='Prompt Gen')
    parser.add_argument('--path', help='Path to raw predictions.', required=True, type=str)
    parser.add_argument('--data_path', help='Path to data file.', required=True, type=str)
    parser.add_argument('--tar_file', help='Path to target file.', required=True, type=str)
    parser.add_argument('--task', help='Task.', required=True, type=str, choices=['nlu', 'pol', 'nlg'])
    args = parser.parse_args()
    args = vars(args)

    return args


def post_process(text):
    text = text.lower()
    text = text.split('[done]', 1)[0].strip()
    text = text.split('[1]', 1)[0].strip()

    return text


def run(args):
    with open(args['path'], 'r') as fp:
        results = json.load(fp)

    parsed_results = []
    statuses = []
    for result in results:
        if args['task'] == 'nlg':
            text = post_process(result)
            parsed_results.append(text)
        else:
            text = post_process(result)
            try:
                parsed_results.append(json.loads(text))
                statuses.append('Ok')
            except:
                parsed_results.append([])
                statuses.append('JSON Error')

    with open(args['data_path'], 'r') as fp:
        data = json.load(fp)

    final = []
    cnt = 0
    for session in data:
        for entry in session:
            final.append({
                'did': entry['dial_id'][4:], 'uid': int(entry['turn_num']),
                'prediction': parsed_results[cnt],
                'nlg': entry['nlg'],
                args['task']: entry[args['task']],
            })
            cnt += 1

    with open(args['tar_file'], 'w') as fp:
        json.dump(final, fp)


if __name__ == '__main__':
    args = read_cli()
    run(args)
