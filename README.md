
## [Understand What LLM Needs: Dual Preference Alignment for Retrieval-Augmented Generation](https://arxiv.org/pdf/2406.18676)</h2>


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/understand-what-llm-needs-dual-preference/knowledge-base-question-answering-on-1)](https://paperswithcode.com/sota/knowledge-base-question-answering-on-1?p=understand-what-llm-needs-dual-preference)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/understand-what-llm-needs-dual-preference/question-answering-on-natural-questions)](https://paperswithcode.com/sota/question-answering-on-natural-questions?p=understand-what-llm-needs-dual-preference)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/understand-what-llm-needs-dual-preference/question-answering-on-triviaqa)](https://paperswithcode.com/sota/question-answering-on-triviaqa?p=understand-what-llm-needs-dual-preference)


*Guanting Dong, Yutao Zhu, Chenghao Zhang, Zechen Wang, Zhicheng Dou and Ji-Rong Wen*

*Gaoling School of Artificial Intelligence, Renmin University of China.*

This is the repository contains core implementations of the **DPA-RAG**, proposed by [Understand What LLM Needs: Dual Preference Alignment for Retrieval-Augmented Generation](https://arxiv.org/abs/2406.18676).


If you find this work helpful for your research, please kindly cite it.


```bibtex
@article{dong2024understand,
  author       = {Guanting Dong and
                  Yutao Zhu and
                  Chenghao Zhang and
                  Zechen Wang and
                  Zhicheng Dou and
                  Ji{-}Rong Wen},
  title        = {Understand What {LLM} Needs: Dual Preference Alignment for Retrieval-Augmented
                  Generation},
  journal      = {CoRR},
  volume       = {abs/2406.18676},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2406.18676},
  doi          = {10.48550/ARXIV.2406.18676},
  eprinttype    = {arXiv},
  eprint       = {2406.18676},
  timestamp    = {Mon, 22 Jul 2024 14:28:34 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2406-18676.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```


---


## 🍯 Overall Framework

<img width="1302" alt="image" src="https://github.com/dongguanting/DPA-RAG/assets/60767110/fde07a6a-fa0d-4099-a6f8-0d16782b7ec4">


**DPA-RAG** is a universal framework for aligning diverse preference knowledge within RAG systems, consisting of three main components:

- **Preference Knowledge Construction:** We design three-step process to identify and synthesize high-quality preference knowledge

- **Reranker-LLM Alignment:** We fine-tune a reranker with multi-grained tasks to achieve alignment between the retriever and LLMs.

- **LLM Self-Alignment:** Before SFT stage, we introduce a pre-alignment phase to help LLMs capture preference-aligned knowledge.


---






## 💻 Data preparation
We design a three-step method to gradually mine, augment, and filter out high-quality preference knowledge:

### 1. Preference Knowledge Constructio
For each samples in different datasets, you need to use [DPR](https://github.com/facebookresearch/DPR) to retrieve the top 100 passages. 

Then, please follow the process of **Preference Knowledge Construction** section to extract documents labeled “Aligned Knowledge” or “Unaligned Knowledge” from different base model.


### 2. Diverse Query Augmentation
Please use **GPT-3.5-turbo** to perform five augmentations for each query, the template and requirements are as follow:
   
![image](https://github.com/dongguanting/DPA-RAG/assets/60767110/cbefab86-74c8-46ea-afc7-5f94b5ca100c)

- Rephrasing. Rephrase the original query with the same intention.
- Complexity. Increase the semantic complexity of the original query.
- Decomposition. Decompose the original query into several sub-problems.
- Constraint. Add more conditional and constrained statements to the original query.
- SPARQL. Rewrite the original query based on the SPARQL syntax and generate it directly

We also provide **1k samples of NQ dataset** for each augmentation, which can be directly downloaded in [here](https://drive.google.com/drive/folders/1fbehvvNzas0VitdBky-pDLDZ_vLSHI81).


### 3. NLI Filtering
Please use use [mDeBERTa](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) as NLI model for consistency filtering between original and augmented samples.

**Data format:**

```json
{
"instruction": "what does jamaican people speak?",
"back_instruction": ["What language do Jamaican people speak?",
"Given their diverse cultural background and official language policies",
"what language do Jamaican people predominantly speak in everyday life as well as in formal settings?",
"What do Jamaicans speak?"]
}
```

**Fliter process:**

```bash
python NLI_filter.py
```


---

## :sparkles: Reranker-LLM Alignment

### 1. Code clone
Run the following command from the command line to clone our code from github to local:

```bash
git clone https://github.com/dongguanting/DPA-RAG.git
```

### 3. Start training

We present some training data and test data. If you need to modify training data and test data, follow a similar data format without missing the necessary fields.

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


## 🌈  LLM Training


For LLM Training，we use the [LlaMA-Factory v0.6.3](https://github.com/hiyouga/LLaMA-Factory/releases/tag/v0.6.3) for Llama2-7B/13B, Mistral-7B, Qwen1.5-0.5B/4B/7B/14B, Phi2-2.7B. Moreover, we use the [LlaMA-Factory v0.8.1](https://github.com/hiyouga/LLaMA-Factory/releases/tag/v0.8.1) for Qwen2-7B, Llama3-8B. Thanks for their excellent work.



### 1. Pre-aligned Training:

We provide the NQ dataset at the prealigned stage [here](https://drive.google.com/drive/folders/1JFCGpnmqfMHGh6X9cMJFtTxVduHkiQXi?usp=sharing). Note that the prealigned data we provide does not include augmented data, please merge augmentation data to unlock more powerful alignment capabilities. You can construct other datasets on your own following our NQ's data format. Please replace the parameters with $ symbols with your own parameters.

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

### 2. SFT Training:

You can find the original training data with top3 passages (w/o data augmentation) [here](https://drive.google.com/drive/folders/1dCCpAVPiwPgjOhuKGcyonwgfr2kntJHZ?usp=sharing).
Please merge your augmentation data to unlock more powerful alignment capabilities.

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

### 3. Inference

You can find our reranked test data with top3 passages [here](https://drive.google.com/drive/folders/1HFAEGX5A5aVFNuWMzA1zLoRtfE-FyoTB?usp=sharing). For WebQSP dataset. we only provide train and test data with top2 passages for aligning.

```bash
 CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
     --stage sft \
     --model_name_or_path $path_to_llama_model \
     --do_predict \
     --flash_attn no \
     --dataset $dataset \
     --template $template \
     --output_dir $OUTPUT_PATH \
     --per_device_eval_batch_size 8 \
     --max_samples 150 \
     --cutoff_len 2048 \
     --predict_with_generate \
     --fp16 \
     --quantization_bit 4 \
     --max_new_tokens 20
```



