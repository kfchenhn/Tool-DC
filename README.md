# Try, Check, and Retry: A Divide-and-Conquer Framework for Boosting Long-Context Tool-Calling Capabilities in LLMs

This repository contains the data-processing, training, and evaluation scripts for **Try, Check, and Retry**, a divide-and-conquer framework for improving the long-context tool-calling capabilities of large language models (LLMs).

## Resources

The released model and dataset are available on Hugging Face:

- **Model:** <a href="https://huggingface.co/kunfeng/Qwen7B-Tool-DC_TB"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="18" height="18" alt="Hugging Face"> Qwen7B-Tool-DC_TB</a>
- **Dataset:** <a href="https://huggingface.co/datasets/kunfeng/Tool-DC_Data"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="18" height="18" alt="Hugging Face"> Tool-DC_Data</a>

## Dataset Preparation

Preprocessed inference and supervised fine-tuning (SFT) datasets are provided in the following directories:

- **Inference datasets:**
  - **BFCL:** `data/bfcl_data`
  - **ACEBench:** `data/acebench_data`
- **SFT training data:** `data/train_data`

## Environment Setup

**Prerequisite:** Python 3.10 or later

### Inference Environment

Install the inference environment with:

```bash
cd code/BFCL
pip install -e .
```

### Training Environment

Install [LlamaFactory](https://github.com/hiyouga/LlamaFactory) with:

```bash
git clone https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
```

## Training

The training configurations are located in:

```text
scripts/tool_sft
```

### Run SFT Training

```bash
llamafactory-cli train scripts/tool_sft/qwen_lora_tool_sft.yaml
```

### Export the Model

```bash
llamafactory-cli export scripts/tool_sft/qwen_lora_merge.yaml
```

## Evaluation

### BFCL

```bash
cd code/BFCL/experiment_code
bash test_w_train_free_method.sh
bash test_w_train_base_method.sh
```

### ACEBench

```bash
cd code/ACEBench/experiment_code
bash test_w_train_free_method.sh
bash test_w_train_base_method.sh
```

## Optional Data Processing

To reproduce the data-generation process, run the following scripts from the repository root.

### Generate Data for Extended Benchmarks

```bash
bash code/data_process_code/get_data_w_more_function/get_data_w_more_function.sh
```

### Generate SFT Training Data

```bash
bash code/data_process_code/get_train_data/get_data_for_train.sh
```
