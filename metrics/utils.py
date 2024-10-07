import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import openai
from time import sleep
from tqdm import tqdm


MAX_RETRIES = 5
MAX_TOKENS = 10
STOP = '\n'

# V2
def get_openai_results(client, prompt, model, n=1):
    ret = openai.ChatCompletion.create(
        engine=model,
        messages=[
            {"role": "system", "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        # stop=STOP,
        temperature=0,
        n=n
    )
    # print(ret)

    return ret


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class PairwiseScorer(object):
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.model.eval()
        self.bsz = 16

        if torch.cuda.device_count() > 0:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model.to(self.device)

    def get_scores(self, gold, pred):
        """
        Input is n gold and m predicted texts.
        Output is a n x m matrix of similarity scores
        """
        all_text = gold + pred
        all_embs = []
        for st in range(0, len(all_text), self.bsz):
            en = st + self.bsz
            texts = all_text[st:en]
            batch = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
            batch = dict([(k, v.to(self.device)) for k, v in batch.items()])

            with torch.no_grad():
                ret = self.model(**batch)

            # Perform pooling
            embs = mean_pooling(ret, batch['attention_mask'])
            # Normalize embeddings
            embs = F.normalize(embs, p=2, dim=1)

            embs = embs.cpu().numpy()
            all_embs.extend(embs)

        gold_embs = np.array(all_embs[:len(gold)])
        pred_embs = np.array(all_embs[len(gold):])
        scores = np.matmul(gold_embs, pred_embs.T)

        return scores


class ChatGPTPairwiseComparator(object):
    def __init__(self):
        openai.api_key = os.environ['OPENAI_API_KEY']
        openai.api_base = os.environ['OPENAI_API_CHECKPOINT'] # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
        openai.api_type = 'azure'
        openai.api_version = '2023-07-01-preview'
        self.model = os.environ['OPENAI_MODEL']

    def zero_shot_wo_dialog(self, s1, s2):
        prompt = """You are an expert in the English language. Your task is identifying whether two phrases have similar meanings. If the two phrases have similar meanings, say "positive." Otherwise, say "negative". Pay special attention to any medical terms present in the phrases. Here are the phrases."""
        prompt += '\n\n'
        prompt += f"Phrase 1: {s1.lower()}\n"
        prompt += f"Phrase 2: {s2.lower()}\n"
        prompt += "Answer (positive/negative):"
    
        return prompt

    def get_scores(self, gold, pred):
        pairs = []
        for gg in gold:
            for pp in pred:
                pairs.append((gg, pp))

        prompts = [self.zero_shot_wo_dialog(s1, s2) for s1, s2 in pairs]
        predictions = []

        for prompt in prompts:
            succeeded = False
            time_to_sleep = 1
            num_retries = 0
            # print(prompt)
            while not succeeded:
                try:
                    ret = get_openai_results(None, prompt, self.model, n=1)
                    succeeded = True
                except openai.error.InvalidRequestError as e:
                    print(e)
                    print('Prompt Error, skipping.')
                    break
                except Exception as e:
                    print(e)
                    print(type(e))
                    print("Failure, retrying!")
                    print("Sleep duration", time_to_sleep)
                    sleep(time_to_sleep)
                    time_to_sleep += 10
                    num_retries += 1
                    if num_retries >= MAX_RETRIES:
                        print('Failed at', idx)
                        exit(0)

            pp = ret.choices[0].message.content
            pp = pp.lower()
            if 'positive' in pp:
                predictions.append(1)
            else:
                predictions.append(0)

        ret = np.zeros((len(gold), len(pred)))
        for ii, ss in enumerate(predictions):
            rr = ii // len(pred)
            cc = ii % len(pred)
            ret[rr, cc] = ss

        return ret


if __name__ == '__main__':
    scorer = PairwiseScorer()
    ret = scorer.get_scores(['random text', 'another random text'], ['random text'])
    print(ret)
