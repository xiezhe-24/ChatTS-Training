# ChatTS-Training
This is a modified version of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) that supports training ChatTS models.

## Requirements
Following the steps in [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
Make sure that `flash-attention` and `DeepSpeed` are installed.

## Usage
1. Put your training data in `data/`.
2. Set your training data path in `data/dataset_info.json`.
3. Configure your base model (see the instructions below), output model, training datasets and training parameters in `scripts/train_chatts.sh`.
4. Run `bash scripts/train_chatts.sh`.

## Instructions for converting base models (Qwen2 Series) to ChatTS format
1. Download the base models (Qwen2 Series) from huggingface
2. Replace `*.py`, `added_tokens.json`, `config.json`, `special_tokens_map.json`, `tokenizer_config.json` in the base model folder with the files in the ChatTS's model (https://huggingface.co/bytedance-research/ChatTS-14B) folder.

## Credit
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
