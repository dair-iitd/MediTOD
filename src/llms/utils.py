import re
import json
import yaml
import argparse

train_tag = 'train'
val_tag = 'valid'
test_tag = 'test'


def read_cli():
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument(
        "-cfg",
        "--config",
        help="Path to config file",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-model_path",
        "--model_path",
        help="Path to checkpoint",
        required=False,
        type=str,
        default=None
    )
    parser.add_argument(
        "-chkpt",
        "--chkpt",
        help="Checkpoint id",
        required=False,
        type=str,
        default=None
    )
    # OVERRIDES
    parser.add_argument(
        "-datapath",
        "--datapath",
        help="Data path to use instead of training",
        required=False,
        type=str,
        default=None
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Learning rate",
        required=False,
        type=float,
        default=None
    )
    parser.add_argument(
        "-fp16",
        "--fp16",
        help="Enable FP16",
        required=False,
        type=str,
        default=None,
        choices=['True', 'False']
    )
    parser.add_argument(
        "-warmup_ratio",
        "--warmup_ratio",
        help="Warm up ratio",
        required=False,
        type=float,
        default=None
    )
    parser.add_argument(
        "-bsz",
        "--batch_size",
        help="Batch size",
        required=False,
        type=int,
        default=None
    )
    parser.add_argument(
        "-seed",
        "--seed",
        help="Seed",
        required=False,
        type=int,
        default=42
    )
    parser.add_argument(
        "-rs",
        "--result_path",
        help="Result path",
        required=False,
        type=str,
        default=None
    )
    args = vars(parser.parse_args())

    return args


def override_config(cfg, ocfg):
    tags = []

    if ocfg['datapath'] is not None:
        cfg['datapath'] = ocfg['datapath']
        tags.append(('datapath', ocfg['datapath']))

    if ocfg['learning_rate'] is not None:
        cfg['train']['learning_rate'] = ocfg['learning_rate']
        tags.append(('learning_rate', ocfg['learning_rate']))

    if ocfg['fp16'] is not None:
        cfg['train']['fp16'] = ocfg['fp16']
        tags.append(('fp16', ocfg['fp16']))

    if ocfg['warmup_ratio'] is not None:
        cfg['train']['warmup_ratio'] = ocfg['warmup_ratio']
        tags.append(('warmup_ratio', ocfg['warmup_ratio']))

    if ocfg['batch_size'] is not None:
        cfg['train']['per_device_train_batch_size'] = ocfg['batch_size']
        tags.append(('batch_size', ocfg['batch_size']))

    if ocfg['seed'] is not None:
        cfg['train']['seed'] = ocfg['seed']
        tags.append(('seed', ocfg['seed']))

    if len(tags) > 0:
        tag = '_'.join([
            f"{k}:{v}" for (k, v) in tags
        ])
        cfg['experiment_name'] = cfg['experiment_name'] + '_' + tag

    return cfg


def get_config(config_file):
    print(f'Reading config from', config_file)
    with open(config_file, 'r') as fp:
        cfg = yaml.safe_load(fp)
    
    return cfg


def get_joint_config(wandb_cfg):
    base_cfg = get_config(wandb_cfg['base_config'])
    base_cfg['base_name'] = base_cfg['experiment_name']

    ov_keys = [base_cfg['experiment_name']]
    for key1 in wandb_cfg.keys():
        if key1 not in base_cfg:
            continue
        for key2 in wandb_cfg[key1].keys():
            base_cfg[key1][key2] = wandb_cfg[key1][key2]
            ov_keys.append(f"{key1}-{key2}:{wandb_cfg[key1][key2]}")
    base_cfg['experiment_name'] = '_'.join(ov_keys)

    return base_cfg


def load_json(fname):
    with open(fname, 'r') as fp:
        obj = json.load(fp)
    return obj


def parse_nlu(response):
    # [action 1] (values) a, b, c (checks) e, f, g [action 2]..
    words = [x.strip() for x in response.split()]
    if len(words) == 0:
        return []

    if words[0] == '<sos_b>':
        words = words[1:]
    if words[-1] == '<eos_b>':
        words = words[:-1]
    if words[0] == '[medical]':
        words = words[1:]

    aidxs = [ii for ii in range(len(words)) if re.search(r"\[([a-zA-Z0-9_\-]+)\]", words[ii]) is not None]
    aidxs.append(len(words))
    actions = []
    for ii, st in enumerate(aidxs[:-1]):
        en = aidxs[ii + 1]
        act = dict()
        act['intent'] = words[st][1:-1]

        vidxs = [jj for jj in range(st + 1, en) if re.search(r"\(([a-zA-Z0-9_]+)\)", words[jj]) is not None]
        vidxs.append(en)

        slots = dict()
        for jj, vst in enumerate(vidxs[:-1]):
            ven = vidxs[jj + 1]
            if words[vst] not in slots:
                slots[words[vst][1:-1]] = []
            vvs = ' '.join(words[vst + 1:ven]).split(',')
            vvs = [x.strip() for x in vvs]
            vvs = [x for x in vvs if len(x) > 0]
            slots[words[vst][1:-1]].extend(vvs)

        if len(slots) > 0:
            act['slots'] = slots
        for kk in list(act):
            if len(act[kk]) == 0:
                del act[kk]
        actions.append(act)

    return actions, True


def parse_pol(response):
    # [action 1] (values) a, b, c (checks) e, f, g [action 2]..
    words = [x.strip() for x in response.split()]

    aidxs = [ii for ii in range(len(words)) if re.search(r"\[([a-zA-Z0-9_\-]+)\]", words[ii]) is not None]
    aidxs.append(len(words))
    actions = []
    for ii, st in enumerate(aidxs[:-1]):
        en = aidxs[ii + 1]
        act = dict()
        act['action'] = words[st][1:-1]

        vidxs = [jj for jj in range(st + 1, en) if re.search(r"\(([a-zA-Z0-9_]+)\)", words[jj]) is not None]
        vidxs.append(en)

        for jj, vst in enumerate(vidxs[:-1]):
            ven = vidxs[jj + 1]
            if words[vst] not in act:
                act[words[vst][1:-1]] = []
            vvs = ' '.join(words[vst + 1:ven]).split(',')
            vvs = [x.strip() for x in vvs]
            vvs = [x for x in vvs if len(x) > 0]
            act[words[vst][1:-1]].extend(vvs)

        for kk in list(act):
            if len(act[kk]) == 0:
                del act[kk]
        actions.append(act)

    return actions, True
