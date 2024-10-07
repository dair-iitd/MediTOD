import os
import torch
import wandb
import numpy as np

from transformers import IntervalStrategy, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM 

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)

from dataset import NluDataset, PolDataset, NlgDataset
from utils import (
    read_cli, train_tag, val_tag,
    get_config, override_config
)
from trainers import NluTrainer, PolTrainer
from transformers import Trainer as HfTrainer


def run(args, report_to='tensorboard'):
    seed = args['train']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    if 'biogpt' in args['model']['wildcard'] or 'medalpaca' in args['model']['wildcard']:
        BASE_MODEL = AutoModelForCausalLM
    else:
        BASE_MODEL = AutoModelForSeq2SeqLM

    if args['train'].get('use_lora', False):
        load_in_8bit = args['train'].get('train_in_4bit', False)
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        if ddp:
            device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        else:
            device_map='auto'

        model = BASE_MODEL.from_pretrained(
            args['model']['wildcard'],
            load_in_8bit=load_in_8bit,
            device_map=device_map
        )
        print(model)
        if load_in_8bit:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=args['train']['lora_r'],
            lora_alpha=args['train']['lora_alpha'],
            # target_modules=('q_proj', 'k_proj', 'v_proj', 'out_proj'),
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print('USING Normal TRAINING')
        model = BASE_MODEL.from_pretrained(args['model']['wildcard'])

    tokenizer = AutoTokenizer.from_pretrained(
        args['model']['wildcard'],
        model_max_length=args['model'].get('model_max_length')
    )

    if args['task'] == 'nlu':
        Dataset = NluDataset
        Trainer = NluTrainer
    elif args['task'] == 'pol':
        Dataset = PolDataset
        Trainer = PolTrainer
    elif args['task'] == 'nlg':
        Dataset = NlgDataset
        Trainer = HfTrainer
        # Using basic trainer instead
    else:
        raise NotImplementedError

    train_dataset = Dataset(
        tag=train_tag,
        mode='train', tokenizer=tokenizer, cfg=args
    )
    val_dataset = Dataset(
        tag=val_tag,
        mode='infer' if args['task'] != 'nlg' else 'train', # Train for NLG as we are using loss for validation
        tokenizer=tokenizer, cfg=args
    )

    print(train_dataset[4]['input_seq'])
    print(train_dataset[4]['output'])
    print(tokenizer.decode(train_dataset[4]['input_ids']))
    # print(tokenizer.decode(train_dataset[4]['labels']))

    train_args = TrainingArguments(
        output_dir=args['destpath'],
        overwrite_output_dir=True,
        remove_unused_columns=False,
        log_level='warning',
        per_device_train_batch_size=args['train']['per_device_train_batch_size'],
        per_device_eval_batch_size=args['dev']['per_device_eval_batch_size'],
        num_train_epochs=args['train']['num_epochs'],
        learning_rate=args['train']['learning_rate'],
        max_steps=args['train'].get('max_steps', -1),
        # save_strategy='epoch',
        save_strategy=args['train'].get('save_strategy', IntervalStrategy.STEPS),
        save_steps=args['train'].get('save_eval_steps', 100),
        seed=args['train']['seed'],
        fp16=args['train']['fp16'],
        tf32=args['train'].get('tf32', False),
        bf16=args['train'].get('bf16', False),
        gradient_checkpointing=args['train'].get('gradient_checkpointing', False),
        # evaluation_strategy="epoch",
        evaluation_strategy=args['train'].get('evaluation_strategy', IntervalStrategy.STEPS),
        eval_steps=args['train'].get('save_eval_steps', 100),
        gradient_accumulation_steps=args['train']['gradient_accumulation_steps'],
        logging_steps=10,
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
    )

    if args['task'] == 'nlg':
        trainer = Trainer(
            model=model, args=train_args,
            train_dataset=train_dataset, eval_dataset=val_dataset,
            data_collator=train_dataset.collate_fn,
            tokenizer=tokenizer
        )
    else:
        trainer = Trainer(
            model=model, args=train_args,
            train_dataset=train_dataset, eval_dataset=val_dataset,
            data_collator=train_dataset.collate_fn,
            cfg=args,
            tokenizer=tokenizer
        )
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
