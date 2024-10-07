from functools import partial
import os
import torch
import wandb
import numpy as np

from transformers import IntervalStrategy, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from peft import LoraConfig, prepare_model_for_kbit_training
from utils import get_bitsandbytes_config, load_data, read_cli, get_config, override_config
from copy import deepcopy


def formatting_prompts_func(batch, tokenizer):
    # Beware: We get batched input only.
    assert isinstance(batch['prompt_input'], list)
    ret = []
    use_template = tokenizer.chat_template is not None
    for ii in range(len(batch['prompt_input'])):
        elements = batch['prompt_input'][ii] + batch['prompt_output'][ii]

        if use_template:
            text = tokenizer.apply_chat_template(elements, add_generation_prompt=False, tokenize=False)
        else:
            text = '\n'.join([x['content'] for x in elements])

        ret.append(text)

    return ret


def run(args, report_to='tensorboard'):
    seed = args['train']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_path = args['model']['wildcard']
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # -----------------------> Model Configuration <----------------------- #
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    else:
        device_map='auto'

    # -----------------------> LoRA config <----------------------- #
    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            'up_proj', 'down_proj', 'gate_proj',
            'k_proj', 'q_proj', 'v_proj', 'o_proj'
        ]
    )
    bnb_config = get_bitsandbytes_config()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        attn_implementation="eager",
    )

    use_template = True
    if 'openbiollm' in args['model']['wildcard'].lower():
        response_template = "[answer]"

    elif 'llama' in args['model']['wildcard'].lower():
        print('WARNING... Not using llama prefix...')
        response_template = "[answer]"
        tokenizer.pad_token = tokenizer.eos_token

    elif 'gemma' in args['model']['wildcard'].lower():
        response_template = '<start_of_turn>model\n'
        tokenizer.padding_side = 'right'

    model = prepare_model_for_kbit_training(model)

    train_dataset = load_data(
        args['datapath'], 'train', args['tasks'], history_size=args.get('history_size', 8),
        system_message=args.get('system_message'),
        ignore_system=args['model'].get('ignore_system', False)
    )
    val_dataset = load_data(
        args['datapath'], 'valid', args['tasks'], history_size=args.get('history_size', 8),
        system_message=args.get('system_message'),
        ignore_system=args['model'].get('ignore_system', False)
    )

    train_args = TrainingArguments(
        output_dir=args['destpath'],
        overwrite_output_dir=True,
        remove_unused_columns=True,
        log_level='debug',
        optim='adamw_bnb_8bit',
        per_device_train_batch_size=args['train']['per_device_train_batch_size'],
        per_device_eval_batch_size=args['dev']['per_device_eval_batch_size'],
        num_train_epochs=args['train']['num_epochs'],
        learning_rate=args['train']['learning_rate'],
        max_steps=args['train'].get('max_steps', -1),
        save_strategy=args['train'].get('save_strategy', IntervalStrategy.STEPS),
        save_steps=args['train'].get('save_eval_steps', 100),
        seed=args['train']['seed'],
        gradient_checkpointing=args['train'].get('gradient_checkpointing', False),
        evaluation_strategy=args['train'].get('evaluation_strategy', IntervalStrategy.STEPS),
        eval_steps=args['train'].get('save_eval_steps', 100),
        gradient_accumulation_steps=args['train']['gradient_accumulation_steps'],
        logging_steps=5,
        ddp_find_unused_parameters=False,
        save_total_limit=args['train'].get('save_total_limit', 5),
        load_best_model_at_end=True,
        metric_for_best_model=args['train'].get('metric_for_best_model'),
        greater_is_better=args['train'].get('greater_is_better'),
        report_to=report_to,
        run_name=args['experiment_name'],
        warmup_ratio=args['train'].get('warmup_ratio', 0.0),
        dataloader_drop_last=True,
        lr_scheduler_type=args['train'].get('lr_scheduler', 'constant'),
        group_by_length=args['train'].get('group_by_length', False)
    )

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer, mlm=False
    )

    func = partial(formatting_prompts_func, tokenizer=tokenizer)
    print(func(train_dataset[10:11])[0])

    trainer = SFTTrainer(
        model=model, args=train_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        tokenizer=tokenizer, data_collator=collator,
        formatting_func=func, peft_config=peft_config,
        max_seq_length=args['model'].get('max_seq_length', 512)
    )

    trainer.model.print_trainable_parameters()
    trainer.train()


if __name__ == "__main__":
    cargs = read_cli()
    args = get_config(cargs['config'])
    args = override_config(args, cargs)

    local_rank = os.environ.get('LOCAL_RANK', '')
    report_to = 'tensorboard'
    if args.get('use_wandb', False):
        import wandb
        wandb.init(
            group=args['experiment_name'],
            name=args['experiment_name'] + local_rank,
            resume='allow'
        )
        report_to='wandb'

    run(args, report_to=report_to)
