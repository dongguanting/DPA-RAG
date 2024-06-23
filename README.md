## Understand What LLM Needs: Dual Preference Alignment for Retrieval-Augmented Generation</h2>

*Guanting Dong, Yutao Zhu, Chenghao Zhang, Zechen Wang, Zhicheng Dou and Ji-Rong Wen*

Gaoling School of Artificial Intelligence, Renmin University of China.

This is the repository contains core implementations of the **D**ual **P**reference **A**lignment for **R**etrieval-**A**ugmented **G**eneration (**DPA-RAG**), proposed by [Understand What LLM Needs: Dual Preference Alignment for Retrieval-Augmented Generation]().


---


## 🍯 Overall Framework

**DPA-RAG** is a universal framework for aligning diverse preference knowledge within RAG systems, consisting of three main components:

**Preference Knowledge Construction:** We identify and synthesize high-quality preference knowledge using five query augmentation strategies and a filtering process.

**Reranker-LLM Alignment:** We fine-tune a reranker with multi-grained alignment tasks, integrating pair-wise, point-wise, and contrastive preference alignment, to achieve external alignment between the retriever and LLMs.

**LLM Self-Alignment:** Prior to the standard SFT stage, we introduce a pre-alignment phase to help LLMs capture preference-aligned knowledge from multiple documents, ensuring internal self-alignment.

<img width="1302" alt="image" src="https://github.com/dongguanting/DPA-RAG/assets/60767110/fde07a6a-fa0d-4099-a6f8-0d16782b7ec4">

---




## :sparkles: Reranker-LLM Alignment

### 1. clone our codes
Run the following command from the command line to clone our code from github to local:

```bash
git clone https://github.com/dongguanting/DPA-RAG.git
```
### 2. prepare data

We present some training data and test data. If you need to modify training data and test data, follow a similar data format without missing the necessary fields.

### 3. start training
```bash
python train_bge_joined.py \
--pretrained_model_path=your/pretrained/bge/path \
--train_data_path=your/train/data/path \
--valid_data_path=your/valid/data/path \
--test_data_path=your/test/data/path \
--gpu=0 \
--outdir=your/output/path \
--tensorboard_log_dir=your/tensorboard/log/path \
--cls_loss \
--rank_loss \
--scl_loss
```

You can choose which GPU to train and test your model on, but right now we don't support multi-GPU training.

You can choose which losses to incorporate when training. For example, if you only want to use classification loss and contrast learning loss, you can run the following command to start training:

```bash
python train_bge_joined.py \
--pretrained_model_path=your/pretrained/bge/path \
--train_data_path=your/train/data/path \
--valid_data_path=your/valid/data/path \
--test_data_path=your/test/data/path \
--gpu=0 \
--outdir=your/output/path \
--tensorboard_log_dir=your/tensorboard/log/path \
--cls_loss \
--scl_loss
```

Tests are automatically performed on the specified data set after the training is complete.

---


## 🎯 LLM Training

For LLM Training， we use the [LlaMA-Factory v0.6.3](https://github.com/hiyouga/LLaMA-Factory/releases/tag/v0.6.3). Thanks for their excellent work.

(1) LLM Self Alignment:


(2) SFT Training:

```bash
deepspeed --num_gpus=8 train_bash.py \
        --deepspeed $deepspeed_zero3_config_path \
        --stage sft \
        --do_train \
        --use_fast_tokenizer \
        --flash_attn \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --model_name_or_path $MODEL_PATH \
        --dataset $dataset \
        --template $Template \
        --finetuning_type full \
        --output_dir $OUTPUT_PATH \
        --overwrite_cache \
        --overwrite_output_dir \
        --warmup_steps 20 \
        --weight_decay 0.1 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --ddp_timeout 9000 \
        --learning_rate 7e-6 \
        --lr_scheduler_type "linear" \
        --logging_steps 1 \
        --cutoff_len 8192 \
        --save_steps 200 \
        --num_train_epochs 3.0 \
        --plot_loss \
        --bf16 
```


