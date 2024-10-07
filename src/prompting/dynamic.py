from commons import task_description, pol_task_description, nlg_task_description
from formating_utils import format_nlu
import os
import json
from copy import deepcopy
import numpy as np

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
from torch.nn.functional import normalize


class DynamicPrompter(object):
    def __init__(
            self, num_exemplars=2, history_size=4,
            dialogs=None, train_ids=None
        ) -> None:
        # print("WARNING............")
        # print("ONLY NLU IS SUPPORTED...............")

        assert history_size >= 2
        assert dialogs is not None and train_ids is not None

        self.exemplars = None
        self.history_size = history_size
        self.model = None
        self.tokenizer = None
        self.documents = None
        self.vectors = None
        self.history_size = history_size
        self.num_exemplars = num_exemplars
        self.device = 'cpu'
        if torch.cuda.device_count() > 0:
            self.device = 'cuda'

        self.setup(dialogs, train_ids)

    def setup(self, dialogs, train_ids):
        self.model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
        self.model.eval()

        documents = []
        for did in train_ids:
            for uid, tuttr in enumerate(dialogs[did]['utterances']):
                if tuttr['speaker'] != 'patient':
                    continue

                uttr = deepcopy(tuttr)
                uttr['did'] = did
                uttr['uid'] = uid

                ft_nlu = format_nlu(uttr['nlu'])
                uttr['formated_nlu'] = ft_nlu

                st = max(0, uid - self.history_size)
                en = uid + 1
                context = [
                    f"{dialogs[did]['utterances'][ii]['speaker']}: {dialogs[did]['utterances'][ii]['text']}"
                    for ii in range(st, en, 1)
                ]

                uttr['dialog_history'] = context[:-2]
                uttr['last_turn'] = context[-2:]

                text = '\n'.join(uttr['dialog_history'])
                prompt = "[dialog history]\n" + text + "\n\n"
                text = '\n'.join(uttr['last_turn'])
                prompt += "[last turn]\n" + text + "\n\n[output]"

                text = json.dumps(ft_nlu)
                output = text 

                uttr['prompt_elements'] = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": output},
                ]

                documents.append(uttr)
        self.documents = documents
                    
        vectors = np.zeros((len(documents), 1024))
        contexts = ['\n'.join(x['last_turn']) for x in documents]
        print(contexts[:2])
        for st in tqdm(range(0, len(contexts), 128)):
            en = min(len(contexts), st + 128)
            tout = self.tokenizer(contexts[st:en], return_tensors='pt', padding=True, truncation=True)
            tout = dict([(k, v.to(self.device)) for k, v in tout.items()])
            with torch.no_grad():
                ret = self.model(**tout)
            embs = ret[0][:, 0]
            embs = normalize(embs, p=2, dim=1)
            vectors[st:en, :] = embs.to("cpu").numpy()
        self.vectors = vectors

        print(f'Total documents', len(self.documents))
        print(f'Total vectors', self.vectors.shape)

    def compute_scores(self, text):
        tout = self.tokenizer([text], return_tensors='pt', padding=True, truncation=True)
        tout = dict([(k, v.to(self.device)) for k, v in tout.items()])
        with torch.no_grad():
            ret = self.model(**tout)

        embs = ret[0][:, 0]
        embs = normalize(embs, p=2, dim=1)
        qvec = embs.to("cpu").numpy()
        scores = np.matmul(self.vectors, qvec.T)[:, 0]

        return scores

    def get_prompt_elements(self, dialog_history, last_turn):
        scores = self.compute_scores('\n'.join(last_turn))
        idxs = np.argsort(scores)[::-1]

        ret = [
            {"role": "system", "content": deepcopy(task_description)},
        ]
        for idx in idxs[:self.num_exemplars]:
            ret.extend(self.documents[idx]['prompt_elements'])

        text = '\n'.join(dialog_history)
        prompt = "[dialog history]\n" + text + "\n\n"
        text = '\n'.join(last_turn)
        prompt += "[last turn]\n" + text + "\n\n[output]"
        ret.append({"role": "user", "content": prompt})

        return ret


class DynamicPolicyPrompter(DynamicPrompter):
    def __init__(self, num_exemplars=2, history_size=4, dialogs=None, train_ids=None) -> None:
        super().__init__(num_exemplars, history_size, dialogs, train_ids)

    def setup(self, dialogs, train_ids):
        print('POLICY PROMPTER')
        self.model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
        self.model.eval()

        documents = []
        for did in train_ids:
            for uid, tuttr in enumerate(dialogs[did]['utterances']):
                if tuttr['speaker'] != 'doctor':
                    continue

                uttr = deepcopy(tuttr)
                uttr['did'] = did
                uttr['uid'] = uid

                st = max(0, uid - self.history_size - 1)
                en = uid
                context = [
                    f"{dialogs[did]['utterances'][ii]['speaker']}: {dialogs[did]['utterances'][ii]['text']}"
                    for ii in range(st, en, 1)
                ]

                uttr['dialog_history'] = context[:-2]
                uttr['last_turn'] = context[-2:]

                if uid == 0:
                    uttr['dialog_state'] = dict()
                else:
                    uttr['dialog_state'] = dialogs[did]['utterances'][uid - 1]['dialog_state']

                prompt = "[dialog state]\n```\n" + json.dumps(uttr['dialog_state'], indent=2) + "\n```\n\n"
                text = '\n'.join(uttr['last_turn'])
                prompt += "[last turn]\n" + text + "\n\n[output]"

                text = json.dumps(uttr['actions'])
                output = text 

                uttr['prompt_elements'] = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": output},
                ]

                documents.append(uttr)
        self.documents = documents
                    
        vectors = np.zeros((len(documents), 1024))
        contexts = ['\n'.join(x['last_turn']) for x in documents]
        print(contexts[:2])
        for st in tqdm(range(0, len(contexts), 128)):
            en = min(len(contexts), st + 128)
            tout = self.tokenizer(contexts[st:en], return_tensors='pt', padding=True, truncation=True)
            tout = dict([(k, v.to(self.device)) for k, v in tout.items()])
            with torch.no_grad():
                ret = self.model(**tout)
            embs = ret[0][:, 0]
            embs = normalize(embs, p=2, dim=1)
            vectors[st:en, :] = embs.to("cpu").numpy()
        self.vectors = vectors

        print(f'Total documents', len(self.documents))
        print(f'Total vectors', self.vectors.shape)

    def get_prompt_elements(self, dialog_state, last_turn):
        scores = self.compute_scores('\n'.join(last_turn))
        idxs = np.argsort(scores)[::-1]

        ret = [
            {"role": "system", "content": deepcopy(pol_task_description)},
        ]
        for idx in idxs[:self.num_exemplars]:
            ret.extend(self.documents[idx]['prompt_elements'])

        prompt = "[dialog state]\n```\n" + json.dumps(dialog_state, indent=2) + "\n```\n\n"
        text = '\n'.join(last_turn)
        prompt += "[last turn]\n" + text + "\n\n[output]"
        ret.append({"role": "user", "content": prompt})

        return ret


class DynamicNLGPrompter(DynamicPrompter):
    def __init__(self, num_exemplars=2, history_size=4, dialogs=None, train_ids=None) -> None:
        super().__init__(num_exemplars, history_size, dialogs, train_ids)

    def setup(self, dialogs, train_ids):
        print('NLG PROMPTER')
        self.model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
        self.model.eval()

        documents = []
        for did in train_ids:
            for uid, tuttr in enumerate(dialogs[did]['utterances']):
                if tuttr['speaker'] != 'doctor':
                    continue

                uttr = deepcopy(tuttr)
                uttr['did'] = did
                uttr['uid'] = uid

                st = max(0, uid - self.history_size - 1)
                en = uid
                context = [
                    f"{dialogs[did]['utterances'][ii]['speaker']}: {dialogs[did]['utterances'][ii]['text']}"
                    for ii in range(st, en, 1)
                ]

                uttr['dialog_history'] = context[:-2]
                uttr['last_turn'] = context[-2:]

                if uid == 0:
                    uttr['actions'] = dict()
                else:
                    uttr['actions'] = dialogs[did]['utterances'][uid]['actions']

                prompt = "[actions]\n```\n" + json.dumps(uttr['actions'], indent=2) + "\n```\n\n"
                text = '\n'.join(uttr['last_turn'])
                prompt += "[last turn]\n" + text + "\n\n[output]"
                output = tuttr['text']

                uttr['prompt_elements'] = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": output},
                ]

                documents.append(uttr)
        self.documents = documents
                    
        vectors = np.zeros((len(documents), 1024))
        contexts = ['\n'.join(x['last_turn']) for x in documents]
        print(contexts[:2])
        for st in tqdm(range(0, len(contexts), 128)):
            en = min(len(contexts), st + 128)
            tout = self.tokenizer(contexts[st:en], return_tensors='pt', padding=True, truncation=True)
            tout = dict([(k, v.to(self.device)) for k, v in tout.items()])
            with torch.no_grad():
                ret = self.model(**tout)
            embs = ret[0][:, 0]
            embs = normalize(embs, p=2, dim=1)
            vectors[st:en, :] = embs.to("cpu").numpy()
        self.vectors = vectors

        print(f'Total documents', len(self.documents))
        print(f'Total vectors', self.vectors.shape)

    def get_prompt_elements(self, actions, last_turn):
        scores = self.compute_scores('\n'.join(last_turn))
        idxs = np.argsort(scores)[::-1]

        ret = [
            {"role": "system", "content": deepcopy(nlg_task_description)},
        ]
        for idx in idxs[:self.num_exemplars]:
            ret.extend(self.documents[idx]['prompt_elements'])

        prompt = "[actions]\n```\n" + json.dumps(actions, indent=2) + "\n```\n\n"
        text = '\n'.join(last_turn)
        prompt += "[last turn]\n" + text + "\n\n[output]"
        ret.append({"role": "user", "content": prompt})

        return ret
