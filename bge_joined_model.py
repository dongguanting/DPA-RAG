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
    def __init__(self, pretrained_model_path, loss_types: List[BgeJoinedModelLoss]):
        assert len(loss_types) > 0
        super(BgeJoinedModel, self).__init__()
        self.loss_types = loss_types
        self.bge = AutoModel.from_pretrained(pretrained_model_path)
        self.classifier = nn.Linear(1024, 2)
        if BgeJoinedModelLoss.ContrastiveLoss in self.loss_types:
            self.scl_loss_func = SupConLoss()
        self.weights_calculator = None

    def forward(self, cls_tokens):
        outputs = self.bge(**cls_tokens).last_hidden_state[:, 0, :]
        outputs = self.classifier(outputs)
        return outputs
