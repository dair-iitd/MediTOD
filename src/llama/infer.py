from functools import partial
import os
import torch
from tqdm import trange
import json
import wandb
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from peft import LoraConfig, prepare_model_for_kbit_training, PeftModelForCausalLM
from utils import get_bitsandbytes_config, load_data, read_cli, get_config, override_config


def formatting_prompts_func(batch, tokenizer):
    # Beware: We get batched input only.
    assert isinstance(batch['prompt_input'], list)
    use_template = tokenizer.chat_template is not None
    ret = []
    for ii in range(len(batch['prompt_input'])):
        elements = batch['prompt_input'][ii]

        if use_template:
            text = tokenizer.apply_chat_template(
                elements, add_generation_prompt=True, tokenize=False
            ) + "[answer]"
        else:
            text = '\n'.join([x['content'] for x in elements]) + '\n[answer]'

        ret.append(text)

    return ret


def run(args, report_to='tensorboard'):
    seed = args['train']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args['model']['wildcard'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # -----------------------> Model Configuration <----------------------- #
    ckpt_path = args['model_path']
    # bnb_config = get_bitsandbytes_config(args['model'].get('quantization'))
    print('Using 8-bit quantization for performance...')
    bnb_config = get_bitsandbytes_config(quantization=8)
    model = AutoModelForCausalLM.from_pretrained(
        args['model']['wildcard'],
        quantization_config=bnb_config,
        attn_implementation='eager',
        use_cache=True
    )
    model = PeftModelForCausalLM.from_pretrained(model, ckpt_path)
    # model = model.merge_and_unload(progressbar=True) # This degrades the performance.

    val_dataset = load_data(
        args['datapath'], args['infer_tag'], args['tasks'], history_size=args.get('history_size', 8),
        system_message=args.get('system_message'),
        ignore_system=args['model'].get('ignore_system', False)
    )
    bsz = 1 # args['dev']['per_device_eval_batch_size']

    # Instantiate the stopping criteria
    gconfig = GenerationConfig(
        num_beams=1, do_sample=False, max_new_tokens=500,
        eos_token_id=tokenizer.convert_tokens_to_ids(tokenizer.eos_token),
        pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    )

    all_responses = []
    for st in trange(0, len(val_dataset), bsz):
        batch = val_dataset[st:st+bsz]
        formatted_prompts = formatting_prompts_func(batch, tokenizer)
        batch = tokenizer(
            formatted_prompts, return_tensors='pt', padding=True, truncation=True,
            max_length=args['model'].get('max_seq_length', 512)
        )
        batch = {k: v.to('cuda') for k, v in batch.items()}

        with torch.no_grad():
            outputs = model.generate(
                **batch, generation_config=gconfig,
                use_cache=True, stop_strings=["[done]", "<|end_header_id|>"],
                tokenizer=tokenizer,
            )
        outputs = outputs.to('cpu')
        input_end = batch['input_ids'].size(1)
        outputs = outputs[:, input_end:]
        responses = tokenizer.batch_decode(outputs)
        if st < 10:
            print(formatted_prompts[0])
            print(responses[0])
            print('-' * 120)
        all_responses.extend(responses)

    with open(args['result_path'], 'w') as fp:
        json.dump(all_responses, fp)


if __name__ == "__main__":
    cargs = read_cli()
    args = get_config(cargs['config'])
    args = override_config(args, cargs)

    local_rank = os.environ.get('LOCAL_RANK', '')
    run(args)
