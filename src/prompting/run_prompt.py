from time import sleep
import openai
import argparse
import json, os
from tqdm import tqdm
from copy import deepcopy

import transformers
import torch

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM


class OpenAIModel(object):
    def __init__(self, model, max_output_len=128) -> None:
        openai.api_key = os.environ['OPENAI_API_KEY']
        openai.api_base = os.environ['OPENAI_API_CHECKPOINT']
        openai.api_type = 'azure'
        openai.api_version = '2023-07-01-preview'
        self.model = model
        self.max_output_len = max_output_len

    def get_results(self, prompt_element):
        backoff = 1
        succeeded = False
        max_tries = 5
        while not succeeded:
            try:
                ret = openai.ChatCompletion.create(
                    engine=self.model,
                    messages=prompt_element,
                    max_tokens=self.max_output_len,
                    # stop=STOP,
                    temperature=0
                )
                succeeded = True
            except Exception as e:
                print(e)
                max_tries -= 1
                print(f'Sleeping for {backoff}s')
                sleep(backoff)
                backoff += 10
                if max_tries == 0:
                    break
        if not succeeded:
            return None
        
        try:
            return ret.choices[0].message.content
        except:
            return None


class Llama3Model(object):
    def __init__(self, model, max_output_len=128) -> None:
        model_id = model

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "quantization_config": quantization_config
            },
            device_map="auto",
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            198, # new line
        ]
        self.max_output_len = max_output_len

    def get_results(self, prompt_element):
        prompt = self.pipeline.tokenizer.apply_chat_template(
            prompt_element, 
            tokenize=False, 
            add_generation_prompt=True
        )

        outputs = self.pipeline(
            prompt,
            max_new_tokens=self.max_output_len,
            # eos_token_id=self.terminators,
            do_sample=False,
            temperature=0.0,
            stop_strings=["<|eot_id|>", self.pipeline.tokenizer.eos_token],
            tokenizer=self.pipeline.tokenizer
        )

        return outputs[0]["generated_text"][len(prompt):].strip()


def read_cli():
    parser = argparse.ArgumentParser(description='OpenAI')
    parser.add_argument(
        "-prompt_file",
        "--prompt_file",
        help="Prompt file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-destpath",
        "--destpath",
        help="Destination path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-model",
        "--model",
        help="Model",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-max_output_len",
        "--max_output_len",
        help="max_output_len",
        required=False,
        type=int,
        default=200,
    )

    args = vars(parser.parse_args())
    return args


def run(args):
    with open(args['prompt_file'], 'r') as fp:
        data = json.load(fp)

    os.makedirs(args['destpath'], exist_ok=True)
    if 'OpenBioLLM' in args['model'] or 'Llama' in args['model']:
        model = Llama3Model(args['model'], max_output_len=args['max_output_len'])
    else:
        model = OpenAIModel(args['model'], max_output_len=args['max_output_len'])

    for ii, entry in enumerate(tqdm(data)):
        tname = os.path.join(args['destpath'], f"{ii}.json")
        if os.path.exists(tname):
            print(f'{ii} already done. Skipping.')
            continue

        prompt_elements = entry['prompt_elements']
        result = model.get_results(prompt_elements)

        tentry = deepcopy(entry)
        tentry['prediction'] = result
        with open(tname, 'w') as fp:
            json.dump(tentry, fp)


if __name__ == '__main__':
    args = read_cli()
    run(args)
