# Try, Check and Retry: A Divide-and-Conquer Framework for Boosting Long-context Tool-Calling Capabilities of LLMs


This is the official implementation of our paper submitted to ACL2026.

## Dataset Preparation

We provide preprocessed Inference and SFT datasets in the following directories:

**Inference Datasets:**

* **BFCL:** ` data/bfcl_data`
* **ACEBench:** ` data/acebench_data`

**SFT Training Dataset:**

* **Training Data:** ` data/train_data`

## Environment Setup

**Prerequisites:** Python >= 3.10

### 1. Inference Environment

To set up the environment for inference:

```bash
cd ./code/bfcl
pip install -e .
```

### 2. Training Environment

```bash
git clone [https://github.com/hiyouga/LlamaFactory.git](https://github.com/hiyouga/LlamaFactory.git)
cd LLaMA-Factory
pip install -e .
```

### Training

The training scripts are located in

```
scripts/tool_sft
```

#### Run SFT Training:

```bash
llamafactory-cli train scripts/tool_sft/qwen_lora_tool_sft.yaml
```

#### Export Model:

```bash
llamafactory-cli export scripts/tool_sft/qwen_lora_merge.yaml
```

### Evaluation

#### BFCL Evaluation

```bash
cd code/BFCL/experiment_code
bash test_w_train_free_method.sh
bash test_w_train_base_method.sh
```

#### ACEBench Evaluation

```bash
cd code/ACEBench/experiment_code
bash test_w_train_free_method.sh
bash test_w_train_base_method.sh
```

### Data Processing (Optional)

If you need to reproduce the data generation process:

#### Get data for extended benchmarks:

```bash
bash code/data_process_code/get_data_w_more_function/get_data_w_more_function.sh
```

### Get data for SFT:

```bash
bash code/data_process_code/get_train_data/get_data_for_train.sh
```
