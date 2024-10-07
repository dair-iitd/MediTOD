import json
import argparse
import tiktoken


def read_cli():
    parser = argparse.ArgumentParser(description='Prompt Estimate')
    parser.add_argument('--prompt_file', help='Path to prompt file.', required=True, type=str)
    parser.add_argument('--model', help='Model name', required=True, type=str)
    parser.add_argument('--max_output_len', help='Max Output Len', required=False, type=int, default=128)
    args = parser.parse_args()
    args = vars(args)

    return args


def run(args):
    with open(args['prompt_file'], 'r') as fp:
        data = json.load(fp)

    input_tokens = 0
    output_tokens = 0
    max_token_size = 0
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    for entry in data:
        prompt = '\n\n'.join([x['content'] for x in entry['prompt_elements']])
        tlen = len(tokenizer.encode(prompt))
        input_tokens += tlen
        output_tokens += args['max_output_len']

        if (tlen + args['max_output_len']) > max_token_size:
            max_token_size = tlen + args['max_output_len']

    if args['model'] == 'gpt-3.5-turbo-0125':
        input_cost = 0.0005
        output_cost = 0.0015
    elif args['model'] == 'gpt-4-turbo-1106-preview':
        input_cost = 0.01
        output_cost = 0.03

    input_cost = (input_tokens / 1000.0) * input_cost
    output_cost = (output_tokens / 1000.0) * output_cost

    print('Total cost', round(input_cost + output_cost, 2))
    print('Max token size', max_token_size)


if __name__ == '__main__':
    args = read_cli()
    run(args)
