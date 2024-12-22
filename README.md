# ChatTS-Training
This is a modified version of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) that supports training ChatTS models.

## Requirements
Following the steps in [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
Make sure that `flash-attention` and `DeepSpeed` are installed.

## Usage
1. Put your training data in `data/`.
2. Set your training data path in `data/dataset_info.json`.
3. Configure your base model, output model, training datasets and training parameters in `scripts/train_chatts.sh`.
4. Run `bash scripts/train_chatts.sh`.

## Credit
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
