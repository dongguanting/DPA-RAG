from enum import Enum, auto
import random
from typing import List
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from sklearn.metrics import f1_score
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from SupContrast.losses import SupConLoss


class WeightsCalculator:
    def __init__(self, model_device) -> None:
        self.device = model_device
        self.MAX_N_ITER = 100
        self.EPS = 1e-3
        self.gamma_model = self.Gamma().to(model_device)

    def reset(self):
        self.gamma_model.set_gamma(0, 1, self.device)

    class Gamma(nn.Module):
        def __init__(self):
            super().__init__()
            self.gamma = nn.parameter.Parameter(torch.linspace(0, 1, 100).unsqueeze(-1))

        def forward(self, M, alpha, e_t):
            temp = (1 - self.gamma) * alpha.expand(100, -1) + self.gamma * e_t.expand(
                100, -1
            )
            loss = temp.matmul(M).matmul(temp.T).diag().sum()
            return loss

        def set_gamma(self, start, end, device):
            self.gamma.data = torch.linspace(start, end, 100).unsqueeze(-1).to(device)

    def search_1d(self, M, alpha, e_t, gamma_model: Gamma):
        loss_rec = 0.0
        start = 0
        end = 1
        for i in range(4):
            gamma_model.set_gamma(start, end, M.device)
            loss = gamma_model(M, alpha, e_t)
            if abs(loss.item() - loss_rec) < self.EPS:
                break
            gamma_model.gamma.grad = None
            loss.backward()
            loss_rec = loss.item()
            gamma = gamma_model.gamma.data
            grad = gamma_model.gamma.grad
            if grad[0] * grad[-1] >= 0:
                return gamma[torch.argmin(abs(grad))].clamp(0, 0.9999)
            try:
                end = gamma[grad > 0].min()
                start = gamma[grad < 0].max()
            except:
                # grad 为 nan ，可能是loss为nan
                raise Exception()
        gamma = gamma_model.gamma.data.mean()
        return gamma

    def calc_weights(self, submodels: List[nn.Module], losses):
        n_loss = len(losses)
        alpha = torch.tensor([1 / n_loss] * n_loss)

        if not losses[0].requires_grad:
            return alpha.to(losses[0].device)

        params = [
            p for model in submodels for p in model.parameters() if p.requires_grad
        ]
        # 构建M矩阵
        g = []
        for loss in losses:
            temp_g = torch.autograd.grad(loss, params, retain_graph=True)
            g.append(torch.cat([i.reshape(-1) for i in temp_g]))
        M = []
        for i in range(n_loss):
            for j in range(n_loss):
                M.append(g[i].matmul(g[j]))
        M = torch.stack(M)
        M = M.reshape(n_loss, n_loss)

        device = M.device
        alpha = alpha.to(device)

        for i in range(self.MAX_N_ITER):
            t = torch.argmin(torch.sum(alpha.expand(n_loss, n_loss) * M, 1))
            e_t = torch.zeros(n_loss).to(device)
            e_t[t] = 1.0
            gamma = self.search_1d(M, alpha, e_t, self.gamma_model)
            alpha = (1 - gamma) * alpha + gamma * e_t
            if gamma < self.EPS or abs(1 - alpha.max()) < self.EPS:
                break

        return alpha


class BgeJoinedModelLoss(Enum):
    ClaasificationLoss = auto()
    RankLoss = auto()
    ContrastiveLoss = auto()


class BgeJoinedModel(nn.Module):
    BGE_MODEL_PATH = "/home/wzc2022/dgt_workspace/LLM-Knowledge-alignment-dgt/bge"

    def __init__(self, loss_types: List[BgeJoinedModelLoss]):
        assert len(loss_types) > 0
        super(BgeJoinedModel, self).__init__()
        self.loss_types = loss_types
        self.bge = AutoModel.from_pretrained(self.BGE_MODEL_PATH)
        self.classifier = nn.Linear(1024, 2)
        if BgeJoinedModelLoss.ContrastiveLoss in self.loss_types:
            self.scl_loss_func = SupConLoss()
        self.weights_calculator = None

    def calc_cls_loss(self, cls_tokens, labels):
        outputs = self.bge(**cls_tokens).last_hidden_state[:, 0, :]
        outputs = self.classifier(outputs)
        loss_cls = nn.functional.cross_entropy(outputs, labels)
        return loss_cls

    def calc_rank_loss(self, rank_tokens, batch_size):
        loss_rank = 0
        outputs = self.bge(**rank_tokens).last_hidden_state[:, 0, :]
        outputs = self.classifier(outputs)[:, 1]
        for group in range(batch_size):
            start = group * 4
            for i in range(start, start + 3):
                for j in range(i + 1, start + 4):
                    loss_rank += -nn.functional.logsigmoid(outputs[i] - outputs[j])
        # C(4,2) * n_groups
        loss_rank /= 6 * batch_size
        return loss_rank

    def calc_scl_loss(self, pos_tokens, neg_tokens, batch_size):
        p_outputs = self.bge(**pos_tokens).last_hidden_state[:, 0, :]
        n_outputs = self.bge(**neg_tokens).last_hidden_state[:, 0, :]
        loss_scl = 0
        for group in range(batch_size):
            features = torch.cat(
                [
                    p_outputs[group * 2 : group * 2 + 2],
                    n_outputs[group * 2 : group * 2 + 2],
                ],
                dim=0,
            ).unsqueeze(1)
            features = nn.functional.normalize(features, dim=-1)
            labels = torch.tensor([1, 1, 0, 0]).to(features.device)
            loss_scl += self.scl_loss_func(features, labels)
        loss_scl /= batch_size
        return loss_scl

    def forward(self, classification, rank, positive, negative):
        """
        输入数据形式：
        {
            "classification": [tokenizer([[query, doc]...]), tensor([label...])]
            "rank": tokenizer([[query, xx]...])
            "positive": tokenizer([[query, positive_sample]...])
            "negative": tokenizer([[query, negative_sample]...])
        }
        """
        # 返回loss
        losses = []
        if BgeJoinedModelLoss.ClaasificationLoss in self.loss_types:
            X, labels = classification
            loss_cls = self.calc_cls_loss(X, labels)
            losses.append(loss_cls)

        if BgeJoinedModelLoss.RankLoss in self.loss_types:
            assert len(rank["input_ids"]) % 4 == 0
            loss_rank = self.calc_rank_loss(
                rank, batch_size=len(rank["input_ids"]) // 4
            )
            losses.append(loss_rank)

        if BgeJoinedModelLoss.ContrastiveLoss in self.loss_types:
            assert (
                len(positive["input_ids"]) == len(negative["input_ids"])
                and len(positive["input_ids"]) % 2 == 0
            )
            loss_scl = self.calc_scl_loss(
                positive, negative, batch_size=len(positive["input_ids"]) // 2
            )
            losses.append(loss_scl)

        if self.weights_calculator is None:
            self.weights_calculator = WeightsCalculator(losses[0].device)
        self.weights_calculator.reset()
        weights = self.weights_calculator.calc_weights(
            [self.bge.embeddings, self.bge.encoder], losses
        )
        loss = torch.stack(losses).matmul(weights)
        return loss
