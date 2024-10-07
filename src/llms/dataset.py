import os
import json
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy


class OsceDataset(object):
    def __init__(
        self, tag, mode, tokenizer, cfg=None
    ):
        assert mode in ['train', 'infer']
        self.raw_data = None
        self.data = None
        self.mode = mode
        self.tokenizer = tokenizer
        self.model_name = cfg['model']['wildcard']
        self.pad_id = tokenizer.pad_token_id
        self.cfg = cfg
        self.prefix_ids = []

        self.load_data(cfg['datapath'], tag)

    def load_data(self, datapath, tag):
        ttag = tag
        if tag == 'valid':
            ttag = 'dev'
        with open(os.path.join(datapath, f"fine-processed-{ttag}.json"), "r") as fp:
            self.raw_data = json.load(fp)

        print(f"Loaded {len(self.raw_data)} dialogs for {tag}...")

    def process_data(self):
        data = []
        for session in self.raw_data:
            context = []
            for entry in session:
                context.append(
                    ('patient', entry['user'].strip("<sos_u>").strip("<eos_u>").strip())
                )
                tmp = deepcopy(entry)
                tmp['context'] = deepcopy(context)
                context.append(
                    ('doctor', entry['resp'].strip("<sos_r>").strip("<eos_r>").strip())
                )
                data.append(tmp)

        self.data = []
        for sample in tqdm(data):
            input_seq, output_seq = self.get_input_output(sample)

            if 'biogpt' in self.model_name or 'medalpaca' in self.model_name:
                input_ids_orig = self.tokenizer(
                    input_seq, add_special_tokens=False
                )['input_ids']
                output_ids = self.tokenizer(output_seq, add_special_tokens=False)['input_ids']

                prefix_ids = deepcopy(self.prefix_ids)
                if 'biogpt' in self.model_name:
                    output_ids.append(self.tokenizer.eos_token_id)
                    prefix_ids.insert(0, 0)

                olen = len(output_ids) if self.mode == 'train' else self.cfg['dev']['max_resp_length']
                tt = len(prefix_ids) + len(input_ids_orig) + olen
                if self.cfg['model']['model_max_length'] < tt:
                    diff = self.cfg['model']['model_max_length'] - len(prefix_ids) - olen
                    input_ids = prefix_ids + input_ids_orig[-diff:]
                else:
                    input_ids = prefix_ids + input_ids_orig

                if self.mode == 'train':
                    labels = [-100 for _ in range(len(input_ids))] + output_ids
                    input_ids = input_ids + output_ids

                attention_mask = [1 for _ in range(len(input_ids))]
                tsample = {
                    'input_seq': input_seq,
                    'input_ids': np.array(input_ids, dtype=np.int64),
                    'attention_mask': np.array(attention_mask, dtype=np.int64),
                    'output': output_seq,
                }
                if self.mode == 'train':
                    tsample['labels'] = np.array(labels, dtype=np.int64)

            elif 't5' in self.model_name:
                input_ids_orig = self.tokenizer(
                    input_seq, add_special_tokens=True
                )['input_ids']
                output_ids = self.tokenizer(output_seq)['input_ids']

                olen = len(output_ids) if self.mode == 'train' else self.cfg['dev']['max_resp_length']
                tt = len(self.prefix_ids) + len(input_ids_orig)

                if tt > self.cfg['model']['model_max_length']:
                    diff = self.cfg['model']['model_max_length'] - len(self.prefix_ids)
                    input_ids = self.prefix_ids + input_ids_orig[-diff:]
                else:
                    input_ids = self.prefix_ids + input_ids_orig
                attention_mask = [1 for _ in range(len(input_ids))]

                tsample = {
                    'input_seq': input_seq,
                    'input_ids': np.array(input_ids, dtype=np.int64),
                    'attention_mask': np.array(attention_mask, dtype=np.int64),
                    'output': output_seq,
                }
                if self.mode == 'train':
                    tsample['labels'] = np.array(output_ids, dtype=np.int64)

            else:
                raise NotImplementedError

            tsample.update(sample)
            self.data.append(tsample)

        print(f"Loaded {len(self.data)} samples for {self.mode}...")
        sizes = [len(x['input_ids']) for x in self.data]
        print(
            'Mean, Min, Max input size',
            np.mean(sizes), np.min(sizes), np.max(sizes)
        )
        if self.mode == 'train':
            sizes = [len(x['labels']) for x in self.data]
            print(
                'Mean, Min, Max output size',
                np.mean(sizes), np.min(sizes), np.max(sizes)
            )                       

    def collate_fn(self, batch):
        max_input_length = -1
        max_output_length = -1
        train_mode = 'labels' in batch[0]

        for entry in batch:
            if max_input_length < len(entry['input_ids']):
                max_input_length = len(entry['input_ids'])

            if train_mode and max_output_length < len(entry['labels']):
                max_output_length = len(entry['labels'])

        assert max_input_length > 0
        assert not train_mode or max_output_length > 0

        bs = len(batch)
        input_token_ids = np.ones((bs,  max_input_length), dtype=np.int64) * self.pad_id
        attention_mask = np.zeros((bs,  max_input_length), dtype=np.int64)

        if train_mode:
            labels = np.ones((bs,  max_output_length), dtype=np.int64) * -100

        for idx, entry in enumerate(batch):
            in_length = len(entry['input_ids'])
            input_token_ids[idx, -in_length:] = entry['input_ids']
            attention_mask[idx, -in_length:] = entry['attention_mask']

            if train_mode:
                out_length = len(entry['labels'])
                labels[idx, -out_length:] = entry['labels']

        ret = {
            "input_ids": input_token_ids,
            "attention_mask": attention_mask,
        }

        if train_mode:
            ret['labels'] = labels

        for k in ret:
            ret[k] = torch.tensor(ret[k])

        return ret

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]


class NluDataset(OsceDataset):
    def __init__(self, tag, mode, tokenizer, cfg=None):
        super().__init__(tag, mode, tokenizer, cfg)

        self.prefix_ids = tokenizer(
            'convert dialog to dialog state:', add_special_tokens=False
        )['input_ids']

        self.process_data()

    def get_input_output(self, sample):
        output = sample['bspn_nlu'].strip('<sos_b>').strip('<eos_b>').strip()

        prompt = []
        for spk, text in sample['context']:
            prompt.append(f"{spk}: {text}")
        prompt = ' '.join(prompt)

        if self.cfg['model'].get('use_trigger', False):
            prompt = prompt + ' answer: '

        return prompt, output


class PolDataset(OsceDataset):
    def __init__(self, tag, mode, tokenizer, cfg=None):
        super().__init__(tag, mode, tokenizer, cfg)

        if self.cfg['model'].get('use_dspn', False):
            self.prefix_ids = tokenizer(
                'convert dialog state and history to dialog act:', add_special_tokens=False
            )['input_ids']
        else:
            self.prefix_ids = tokenizer(
                'convert history to dialog act:', add_special_tokens=False
            )['input_ids']

        self.process_data()

    def get_input_output(self, sample):
        output = sample['aspn'].strip('<sos_a>').strip('<eos_a>').strip()

        if self.cfg['model'].get('use_dspn', False):
            history = []
            for spk, text in sample['context'][-4:]:
                history.append(f"{spk}: {text}")
            history = ' '.join(history)
            dst = sample['bspn'].strip('<sos_b>').strip('<eos_b>').strip()

            prompt = f'dst: {dst} history: {history}'
        else:
            history = []
            for spk, text in sample['context'][:]:
                history.append(f"{spk}: {text}")
            history = ' '.join(history)
            prompt = f'history: {history}'

        if self.cfg['model'].get('use_trigger', False):
            prompt = prompt + ' answer: '

        return prompt, output


class NlgDataset(OsceDataset):
    def __init__(self, tag, mode, tokenizer, cfg=None):
        super().__init__(tag, mode, tokenizer, cfg)

        if self.cfg['model'].get('use_aspn', False):
            self.prefix_ids = tokenizer(
                'convert dialog actions and history to dialog act:', add_special_tokens=False
            )['input_ids']
        else:
            self.prefix_ids = tokenizer(
                'convert history to doctor response:', add_special_tokens=False
            )['input_ids']

        self.process_data()

    def get_input_output(self, sample):
        output = sample['resp'].strip('<sos_r>').strip('<eos_r>').strip()

        if self.cfg['model'].get('use_dspn', False):
            history = []
            for spk, text in sample['context'][-4:]:
                history.append(f"{spk}: {text}")
            history = ' '.join(history)
            actions = sample['aspn'].strip('<sos_a>').strip('<eos_a>').strip()

            prompt = f'actions: {actions} history: {history}'
        else:
            history = []
            for spk, text in sample['context']:
                history.append(f"{spk}: {text}")
            history = ' '.join(history)
            prompt = f'history: {history}'

        if self.cfg['model'].get('use_trigger', False):
            prompt = prompt + ' answer: '

        return prompt, output
