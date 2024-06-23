import random
import os
from typing import Dict
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

from joined_dataset import JoinedDataset, Collater
from bge_joined_model import BgeJoinedModel, BgeJoinedModelLoss, WeightsCalculator


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        loss_types,
        optimizer,
        lr_scheduler,
        device,
        writer,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.loss_types = loss_types
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.writer = writer
        self.weights_calculator = WeightsCalculator(self.device)

    def calc_cls_loss(self, cls_tokens, labels):
        outputs = self.model.bge(**cls_tokens).last_hidden_state[:, 0, :]
        outputs = self.model.classifier(outputs)
        loss_cls = nn.functional.cross_entropy(outputs, labels)
        return loss_cls

    def calc_rank_loss(self, rank_tokens, batch_size):
        loss_rank = 0
        outputs = self.model.bge(**rank_tokens).last_hidden_state[:, 0, :]
        outputs = self.model.classifier(outputs)[:, 1]
        for group in range(batch_size):
            start = group * 4
            for i in range(start, start + 3):
                for j in range(i + 1, start + 4):
                    loss_rank += -nn.functional.logsigmoid(outputs[i] - outputs[j])
        # C(4,2) * n_groups
        loss_rank /= 6 * batch_size
        return loss_rank

    def calc_scl_loss(self, pos_tokens, neg_tokens, batch_size):
        p_outputs = self.model.bge(**pos_tokens).last_hidden_state[:, 0, :]
        n_outputs = self.model.bge(**neg_tokens).last_hidden_state[:, 0, :]
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
            loss_scl += self.model.scl_loss_func(features, labels)
        loss_scl /= batch_size
        return loss_scl

    def cal_loss(self, classification, rank, positive, negative):
        """
        输入数据形式：
        {
            "classification": [tokenizer([[query, doc]...]), tensor([label...])]
            "rank": tokenizer([[query, xx]...])
            "positive": tokenizer([[query, positive_sample]...])
            "negative": tokenizer([[query, negative_sample]...])
        }
        """
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
                positive,
                negative,
                batch_size=len(positive["input_ids"]) // 2,
            )
            losses.append(loss_scl)

        self.weights_calculator.reset()
        weights = self.weights_calculator.calc_weights(
            [self.model.bge.embeddings, self.model.bge.encoder], losses
        )
        loss = torch.stack(losses).matmul(weights)
        return loss

    def train_loop(self, epoch, total_loss):
        progress_bar = tqdm(range(len(self.train_dataloader)))
        progress_bar.set_description(f"loss: {0:>7f}")
        finish_step_num = (epoch - 1) * len(self.train_dataloader)

        self.model.train()
        for step, sample in enumerate(self.train_dataloader, start=1):
            for k, v in sample.items():
                if isinstance(v, list):
                    sample[k] = [vi.to(self.device) for vi in v]
                else:
                    sample[k] = v.to(self.device)

            loss = self.cal_loss(**sample)
            self.writer.add_scalar("loss", loss, step + finish_step_num)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            total_loss += loss.item()
            progress_bar.set_description(
                f"loss: {total_loss/(finish_step_num + step):>7f}"
            )
            progress_bar.update(1)
        return total_loss

    def test_loop(self, dataloader, dataset_type="Test"):
        assert dataset_type in ["Train", "Valid", "Test"]

        self.model.eval()
        tp, fp, fn = 0, 0, 0
        y0 = y1 = 0
        with torch.no_grad():
            for sample in dataloader:
                X, y = sample["classification"] if dataset_type == "Train" else sample
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X).argmax(1)
                tp += torch.sum((pred == 1) & (y == 1)).item()
                fp += torch.sum((pred == 1) & (y == 0)).item()
                fn += torch.sum((pred == 0) & (y == 1)).item()
                y0 += torch.sum(pred == 0).item()
                y1 += torch.sum(pred == 1).item()
        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        except ZeroDivisionError as e:
            print(e)
            print(f"n_y == 0: {y0}\nn_y == 1: {y1}")
            return 0
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"{dataset_type} dataset precision: {(100*precision):>0.1f}%")
        print(f"{dataset_type} dataset recall: {(100*recall):>0.1f}%")
        print(f"{dataset_type} dataset f1: {(100*f1):>0.1f}%")
        print(f"n_y == 0: {y0}\nn_y == 1: {y1}")
        return f1

    def train(self, epoch_num, outdir):
        total_loss = 0.0
        best_f1 = 0.0
        os.makedirs(outdir, exist_ok=True)
        for t in range(epoch_num):
            print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
            total_loss = self.train_loop(t + 1, total_loss)
            train_f1 = self.test_loop(self.train_dataloader, dataset_type="Train")
            self.writer.add_scalar("f1/train_acc", train_f1, t + 1)
            valid_f1 = self.test_loop(self.valid_dataloader, dataset_type="Valid")
            self.writer.add_scalar("f1/valid_f1", valid_f1, t + 1)
            if valid_f1 > best_f1:
                best_f1 = valid_f1
                print("saving new weights...\n")
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        outdir,
                        f"epoch_{t+1}_valid_f1_{(100*valid_f1):0.1f}_model_weights.bin",
                    ),
                )
        self.test_loop(self.test_dataloader, dataset_type="Test")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--valid_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--gpu", type=str, choices=["0", "1"], default="0")
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--tensorboard_log_dir", type=str)
    parser.add_argument("--cls_loss", action="store_true")
    parser.add_argument("--rank_loss", action="store_true")
    parser.add_argument("--scl_loss", action="store_true")

    args = parser.parse_args()
    print(f"Using device: gpu:{args.gpu}")

    return args


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def main():
    args = get_args()
    seed_everything(42)
    learning_rate = 1e-5
    batch_size = 4
    epoch_num = 10
    writer = SummaryWriter(args.tensorboard_log_dir)
    device = f"cuda:{args.gpu}"

    checkpoint = args.pretrained_model_path
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=512)

    train_data = JoinedDataset(args.train_data_path)
    collater = Collater(tokenizer, is_train=True)
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collater
    )

    valid_data = JoinedDataset(args.valid_data_path)
    collater = Collater(tokenizer, is_train=False)
    valid_dataloader = DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, collate_fn=collater
    )

    test_data = JoinedDataset(args.valid_data_path)
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=collater
    )
    loss_types = []
    if args.cls_loss:
        loss_types.append(BgeJoinedModelLoss.ClaasificationLoss)
    if args.rank_loss:
        loss_types.append(BgeJoinedModelLoss.RankLoss)
    if args.scl_loss:
        loss_types.append(BgeJoinedModelLoss.ContrastiveLoss)
    model = BgeJoinedModel(checkpoint, loss_types)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = None
    # lr_scheduler = get_scheduler(
    #     "linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=epoch_num * len(train_dataloader),
    # )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        loss_types=loss_types,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        writer=writer,
    )

    trainer.train(epoch_num=epoch_num, outdir=args.outdir)
    print("Done!")


if __name__ == "__main__":
    main()
