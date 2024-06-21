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
from bge_joined_model import BgeJoinedModel, BgeJoinedModelLoss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str)
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


def train_loop(
    dataloader,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    total_loss,
    writer,
    device,
):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f"loss: {0:>7f}")
    finish_step_num = (epoch - 1) * len(dataloader)

    model.train()
    for step, sample in enumerate(dataloader, start=1):
        for k, v in sample.items():
            if isinstance(v, list):
                sample[k] = [vi.to(device) for vi in v]
            else:
                sample[k] = v.to(device)
        loss = model(**sample)
        writer.add_scalar("loss", loss, step + finish_step_num)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f"loss: {total_loss/(finish_step_num + step):>7f}")
        progress_bar.update(1)
    return total_loss


# def test_loop(dataloader, model, device, mode="Test"):
#     assert mode in ["Valid", "Test"]

#     model.eval()
#     tp, fp, fn = 0, 0, 0
#     y0 = y1 = 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X).argmax(1)
#             tp += torch.sum((pred == 1) & (y == 1)).item()
#             fp += torch.sum((pred == 1) & (y == 0)).item()
#             fn += torch.sum((pred == 0) & (y == 1)).item()
#             y0 += torch.sum(pred == 0).item()
#             y1 += torch.sum(pred == 1).item()
#     try:
#         precision = tp / (tp + fp)
#         recall = tp / (tp + fn)
#     except ZeroDivisionError as e:
#         print(e)
#         print(f"n_y == 0: {y0}\nn_y == 1: {y1}")
#         return 0
#     f1 = 2 * (precision * recall) / (precision + recall)
#     print(f"{mode} precision: {(100*precision):>0.1f}%\n")
#     print(f"{mode} recall: {(100*recall):>0.1f}%\n")
#     print(f"{mode} f1: {(100*f1):>0.1f}%\n")
#     print(f"n_y == 0: {y0}\nn_y == 1: {y1}")
#     return f1


def main():
    args = get_args()
    seed_everything(42)
    learning_rate = 1e-5
    batch_size = 4
    epoch_num = 10
    writer = SummaryWriter(args.tensorboard_log_dir)
    device = f"cuda:{args.gpu}"

    checkpoint = "/home/wzc2022/dgt_workspace/LLM-Knowledge-alignment-dgt/bge"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=512)

    train_data = JoinedDataset(args.train_data_path)
    collater = Collater(tokenizer)
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collater
    )

    # valid_data = JoinedDataset(
    #     "/home/wzc2022/dgt_workspace/LLM-Knowledge-alignment-dgt/data/roberta_data/2classes/test.jsonl"
    # )
    # valid_dataloader = DataLoader(
    #     valid_data, batch_size=batch_size, shuffle=False, collate_fn=collater
    # )
    loss_types = []
    if args.cls_loss:
        loss_types.append(BgeJoinedModelLoss.ClaasificationLoss)
    if args.rank_loss:
        loss_types.append(BgeJoinedModelLoss.RankLoss)
    if args.scl_loss:
        loss_types.append(BgeJoinedModelLoss.ContrastiveLoss)
    model = BgeJoinedModel(loss_types)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = None
    # lr_scheduler = get_scheduler(
    #     "linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=epoch_num * len(train_dataloader),
    # )

    total_loss = 0.0
    best_f1 = 0.0
    for t in range(epoch_num):
        print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
        total_loss = train_loop(
            train_dataloader,
            model,
            optimizer,
            lr_scheduler,
            t + 1,
            total_loss,
            writer,
            device,
        )
        # train_f1 = test_loop(train_dataloader, model, device, mode="Valid")
        # writer.add_scalar("f1/train_acc", train_f1, t + 1)
        # valid_f1 = test_loop(valid_dataloader, model, device, mode="Valid")
        # writer.add_scalar("f1/valid_f1", valid_f1, t + 1)
        # if valid_f1 > best_f1:
        #     best_f1 = valid_f1
        #     print("saving new weights...\n")
        #     torch.save(
        #         model.state_dict(),
        #         os.path.join(
        #             args.outdir,
        #             f"epoch_{t+1}_valid_f1_{(100*valid_f1):0.1f}_model_weights.bin",
        #         ),
        #     )
    print("Done!")


if __name__ == "__main__":
    main()
