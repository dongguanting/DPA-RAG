import random
import os
import jsonlines
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AutoModelForSequenceClassification,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
)


class JoinedDataset(Dataset):

    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        idx = 0
        with jsonlines.open(data_file, "r") as fin:
            for line in fin.iter():
                Data[idx] = line
                idx += 1
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self.data[i] for i in range(start, stop, step)]
        return self.data[idx]


class Collater:
    def __init__(self, tokenizer, is_train) -> None:
        self._tokenizer = tokenizer
        self.is_train = is_train

    def tokenize(self, *args):
        return self._tokenizer(
            *args,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def collate_training_data(self, batch_samples):
        """
        输入数据形式：
        [
            {
                "id": xx,
                "query": xx,
                "classification":{"text": xx, "label": xx},
                "rank_list":[xx, xx, xx, xx],
                "positive": [xx...],
                "negative": [xx...],
            }...
        ]

        输出数据形式：
        {
            "classification": [tokenizer([[query, doc]...]), tensor([label...])]
            "rank": tokenizer([[query, xx]...])
            "positive": tokenizer([[query, positive_sample]...])
            "negative": tokenizer([[query, negative_sample]...])
        }
        """
        res = {
            "classification": [
                self.tokenize(
                    [
                        [sample["query"], sample["classification"]["text"]]
                        for sample in batch_samples
                    ]
                ),
                # 1->1, 2->0
                torch.tensor(
                    [2 - sample["classification"]["label"] for sample in batch_samples]
                ),
            ],
            "rank": self.tokenize(
                [
                    [sample["query"], doc]
                    for sample in batch_samples
                    for doc in sample["rank_list"]
                ]
            ),
            "positive": self.tokenize(
                [
                    [sample["query"], pos_doc]
                    for sample in batch_samples
                    for pos_doc in sample["positive"]
                ]
            ),
            "negative": self.tokenize(
                [
                    [sample["query"], neg_doc]
                    for sample in batch_samples
                    for neg_doc in sample["negative"]
                ]
            ),
        }
        return res

    def collate_valid_and_test_data(self, batch_samples):
        """
        输入数据形式：
        [
           {"question":"", "ctx":"", "category": xx}
        ]
        输出数据形式：
        tokenizer([[query, doc]...]), tensor([label...])
        """
        return (
            self.tokenize(
                [[sample["question"], sample["ctx"]] for sample in batch_samples]
            ),
            torch.tensor([2 - sample["category"] for sample in batch_samples]),
        )

    def __call__(self, batch_samples):
        if self.is_train:
            return self.collate_training_data(batch_samples)
        else:
            return self.collate_valid_and_test_data(batch_samples)
