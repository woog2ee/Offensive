import torch
from torch.utils.data import Dataset
import json
from tqdm import tqdm
from operator import itemgetter


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        infos = [[dict_['text'], dict_['label']] for dict_ in data]
        texts = list(map(itemgetter(0), infos))
        self.labels = list(map(itemgetter(1), infos))

        self.tokenized = [tokenizer(text, padding='max_length',
                                    truncation=True, max_length=max_len)['input_ids']
                          for text in tqdm(texts)]

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, idx):
        input_ids, label = self.tokenized[idx], self.labels[idx]
        item = {'input_ids': input_ids, 'label': label}
        return {k: torch.tensor(v) for k, v in item.items()}