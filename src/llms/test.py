import os
import numpy as np
import torch

from dataset import NluDataset, PolDataset, NlgDataset
from utils import (
    read_cli, train_tag, val_tag, test_tag,
    get_config, override_config
)
from peft import PeftModel

from trainers import NluTrainer, PolTrainer, NlgTrainer

from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

os.environ["WANDB_DISABLED"] = "true"


def run(args):
    seed = args['train']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    chkpt_path = os.path.join(args['destpath'], f"checkpoint-{args['chkpt']}")
    if 'biogpt' in args['model']['wildcard'] or 'medalpaca' in args['model']['wildcard']:
        BASE_MODEL = AutoModelForCausalLM
    else:
        BASE_MODEL = AutoModelForSeq2SeqLM

    if args['train']['use_lora']:
        model = BASE_MODEL.from_pretrained(
            args['model']['wildcard'],
            load_in_4bit=args['train'].get('train_in_4bit', False)
        )
        print(model)
        model = PeftModel.from_pretrained(model, model_id=chkpt_path)
    else:
        model = BASE_MODEL.from_pretrained(chkpt_path)

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
        Trainer = NlgTrainer
    else:
        raise NotImplementedError

    val_dataset = Dataset(
        tag=test_tag,
        mode='infer', tokenizer=tokenizer, cfg=args
    )

    train_args = TrainingArguments(
        output_dir=args['destpath'],
        overwrite_output_dir=True,
        per_device_train_batch_size=args['train']['per_device_train_batch_size'],
        per_device_eval_batch_size=args['dev']['per_device_eval_batch_size'],
    )

    trainer = Trainer(
        cfg=args,
        tokenizer=tokenizer,
        model=model, args=train_args,
        # train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=val_dataset.collate_fn,
    )
    # metrics = trainer.evaluate(train_dataset)
    metrics = trainer.evaluate(
        val_dataset, save_results=True,
        result_path=args['result_path']
    )
    print(metrics)


if __name__ == "__main__":
    cargs = read_cli()

    model_path = cargs['model_path']
    cfg_path = cargs['config']
    args = get_config(cfg_path)
    args['destpath'] = model_path
    if cargs['datapath'] is not None:
        args['datapath'] = cargs['datapath']
    else:
        del cargs['datapath']

    if cargs['batch_size'] is not None:
        args['dev']['per_device_eval_batch_size'] = cargs['batch_size']
    else:
        del cargs['batch_size']

    args.update(cargs)
    run(args)
