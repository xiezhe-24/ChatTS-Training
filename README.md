# ChatTS-Training
This is a modified version of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) that supports training ChatTS models.

## News
- **2025/08/01**: We have updated the code for data preprocessing. No need to preprocess the dataset before training now!
    - Please download the **latest datasets** (we have updated them) from [ChatTS-Training-Dataset](https://huggingface.co/datasets/ChatTSRepo/ChatTS-Training-Dataset).
    - If you want to generate the datasets by yourself, please use `no` encoding instead of `sp` encoding when generating the data. 

## Requirements
Following the steps in [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
Make sure that `flash-attention` and `DeepSpeed` are installed.

## Usage
1. Put your training data in `data/`.
2. Set your training data path in `data/dataset_info.json`.
3. Configure your base model (see the instructions below), output model, training datasets and training parameters in `scripts/train_chatts.sh`.
4. Run `bash scripts/train_chatts.sh` for full SFT. Run `bash scripts/train_lora.sh` for LoRA SFT.

## Instructions for converting base models (Qwen2 Series) to ChatTS format
1. Download the base models (Qwen2 Series) from huggingface
2. Replace `*.py`, `added_tokens.json`, `config.json`, `special_tokens_map.json`, `tokenizer_config.json` in the base model folder with the files in the ChatTS's model (https://huggingface.co/bytedance-research/ChatTS-14B) folder.

## Credit
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
