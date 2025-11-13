# ChatTS-Training
This is a modified version of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) that supports training ChatTS models.

## News
- **2025/11/04**: LoRa and Qwen3 training is now supported.
- **2025/08/01**: We have updated the code for data preprocessing. No need to preprocess the dataset before training now!
    - Please download the **latest datasets** (we have updated them) from [ChatTS-Training-Dataset](https://huggingface.co/datasets/ChatTSRepo/ChatTS-Training-Dataset).
    - If you want to generate the datasets by yourself, please use `no` encoding instead of `sp` encoding when generating the data. 

## Requirements
Following the steps in [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
Make sure that `flash-attention` and `DeepSpeed` are installed.

## Instructions for converting Qwen base models to ChatTS format
1. Download the base models ([Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) or [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)) from huggingface
2. Replace `*.py`, `added_tokens.json`, `config.json`, `special_tokens_map.json`, `tokenizer_config.json` in the base model folder with the files in the[ChatTS-8B](https://huggingface.co/bytedance-research/ChatTS-8B) or [ChatTS-14B](https://huggingface.co/bytedance-research/ChatTS-14B) repo according to type of your base model.
3. **Initialization:** To ensure training stability, we strongly recommend using Xavier normal initialization for the parameters of `ts_encoder`. You can first load the model created in the previous steps using `AutoModelForCausalLM.from_pretrained` in Python, then apply Xavier normal initialization to the `model.ts_encoder` part, and finally save the model using `save_pretrained`. For detailed API usage, please refer to the [official Transformers documentation](https://huggingface.co/docs/transformers/en/index).


## Steps to reproduce
1. Download the training datasets from [ChatTS-Training-Dataset](https://huggingface.co/datasets/ChatTSRepo/ChatTS-Training-Dataset). Put the folders under `data/` (e.g., `data/align_256/`, `data/align_random/`, etc).
2. Configure your base model (see the instructions below), output model, training datasets and training parameters in `scripts/full/train_stage1.sh` and `train_stage2.sh`.
3. Run `bash scripts/train_stage1.sh` and `bash scripts/train_stage2.sh`.

## Use your own datasets
1. **If you want to use your own datasets**, put your own training data in `data/`. The example of dataset format is shown in [chatts_dev.jsonl](data/chatts_dev.jsonl). Set your training data path in `data/dataset_info.json`.
2. Configure your base model (see the instructions below), output model, training datasets and training parameters in `scripts/full/dev.sh` (for full SFT) or `scripts/lora/dev.sh`.
3. Run `bash scripts/train_chatts.sh` for full SFT. Run `bash scripts/train_lora.sh` for LoRA SFT.

## Credit
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
