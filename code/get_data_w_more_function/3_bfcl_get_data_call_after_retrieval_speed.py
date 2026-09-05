import argparse
import os
import json
import csv
import random
import copy
from typing import List, Dict, Any, Set, Tuple
from string import Template
from collections import defaultdict

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

PROMPT_TEMPLATE = Template("""
Carefully analyze the functionality of the two functions below. Ignore differences in syntax, variable names, code style, or implementation details. Focus solely on what the functions logically do.

- If the two functions perform completely unrelated tasks, output: 0 
- If the two functions have partially overlapping functionality (i.e., they share some common purpose or behavior, but are not identical), output: 1 
- If the two functions are functionally equivalent, output: 2 

Respond only with a single digit: 0, 1, or 2. Do not include any other text, explanation, or formatting.

Function 1:
$function1

Function 2:
$function2
""")

def read_tsv_as_dicts(path: str, encoding: str = "utf-8") -> Dict[str, str]:
    corpus_dict = {}
    if not os.path.exists(path):
        return corpus_dict
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            corpus_dict[str(row["docid"])] = row["document_content"]
    return corpus_dict

def load_jsonl(path: str) -> List[Dict]:
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def process_subset(
    subset_name: str,
    data_dir: str,
    target_base_dir: str,
    ori_data_base_dir: str,
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer: Any,
    target_nums: List[int],
    cand_limit: int = 60
):
    subdir = os.path.join(data_dir, subset_name)
    max_fun_num = max(target_nums)

    all_exist = True
    for num in target_nums:
        target_dir_n = f"{target_base_dir}_{num}"
        target_file_n = os.path.join(target_dir_n, subset_name + ".json")
        if not os.path.exists(target_file_n):
            all_exist = False
            break
    
    if all_exist:
        print(f"Skipping {subset_name}, all target files exist.")
        return

    print(f"******** Processing {subset_name} for targets: {target_nums} ***********")

    retr_corpus_id_path = os.path.join(subdir, "ori/ToolRetriever/test_retr_corpus_id.json")
    corpus_path = os.path.join(subdir, "ori/corpus.tsv")
    ori_data_path = os.path.join(ori_data_base_dir, subset_name + ".json")

    if not os.path.exists(corpus_path) or not os.path.exists(ori_data_path) or not os.path.exists(retr_corpus_id_path):
        print(f"Missing files for {subset_name}, skipping.")
        return
        
    corpus_dict = read_tsv_as_dicts(corpus_path)
    ori_datalist = load_jsonl(ori_data_path)
    with open(retr_corpus_id_path, "r", encoding="utf-8") as f:
        retr_corpus_id_list = json.load(f)

    all_prompts = []
    prompt_metadata: List[Tuple[int, int]] = []
    all_candidates_map: Dict[int, List[Dict]] = {}

    for idx, data_item in enumerate(tqdm(ori_datalist, desc=f"Building Prompts [{subset_name}]")):
        ori_fun_list = data_item.get("function", [])
        if len(ori_fun_list) >= max_fun_num:
            continue

        try:
            corpus_id_list = retr_corpus_id_list[idx]["corpus_id_list"]
        except (IndexError, KeyError):
            continue

        candidate_funs = []
        for cid in corpus_id_list:
            fun_str = corpus_dict.get(str(cid))
            if fun_str:
                try:
                    candidate_funs.append(json.loads(fun_str))
                except json.JSONDecodeError:
                    continue
        
        candidate_funs = candidate_funs[:cand_limit]
        all_candidates_map[idx] = candidate_funs
        original_fun_names = {f["name"] for f in ori_fun_list}

        for cand_idx, cand_fun in enumerate(candidate_funs):
            if cand_fun["name"] in original_fun_names:
                continue

            for ori_fun in ori_fun_list:
                user_message = PROMPT_TEMPLATE.substitute(
                    function1=json.dumps(cand_fun, indent=4),
                    function2=json.dumps(ori_fun, indent=4),
                )
                messages = [{"role": "user", "content": user_message}]
                text_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                all_prompts.append(text_prompt)
                prompt_metadata.append((idx, cand_idx))

    bad_candidates = defaultdict(set)
    if all_prompts:
        print(f"Running inference on {len(all_prompts)} prompts...")
        outputs = llm.generate(all_prompts, sampling_params)
        for i, output in enumerate(outputs):
            res_text = output.outputs[0].text.strip()
            d_idx, c_idx = prompt_metadata[i]
            if res_text in ["1", "2"]:
                bad_candidates[d_idx].add(c_idx)

    for target_num in target_nums:
        target_dir_n = f"{target_base_dir}_{target_num}"
        os.makedirs(target_dir_n, exist_ok=True)
        target_file_n = os.path.join(target_dir_n, subset_name + ".json")
        
        if os.path.exists(target_file_n):
            continue
            
        new_datalist = []
        for idx, data_item in enumerate(ori_datalist):
            current_item = copy.deepcopy(data_item)
            ori_fun_list = current_item.get("function", [])
            
            if idx in all_candidates_map and len(ori_fun_list) < target_num:
                candidate_funs = all_candidates_map[idx]
                bad_indices_for_item = bad_candidates[idx]
                current_names = {f["name"] for f in ori_fun_list}
                
                for cand_idx, cand_fun in enumerate(candidate_funs):
                    if len(ori_fun_list) >= target_num:
                        break
                    if cand_idx not in bad_indices_for_item and cand_fun["name"] not in current_names:
                        ori_fun_list.append(cand_fun)
                        current_names.add(cand_fun["name"])
            
            current_item["function"] = ori_fun_list
            random.Random(42).shuffle(current_item["function"])
            new_datalist.append(current_item)
        
        save_jsonl(new_datalist, target_file_n)

def main():
    parser = argparse.ArgumentParser(description="Filter and augment BFCL/AceBench data using vLLM.")
    
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory for retrieved data subsets.")
    parser.add_argument("--target_dir", type=str, required=True, help="Output directory base path (will append _num).")
    parser.add_argument("--ori_data_dir", type=str, required=True, help="Directory containing original JSON files.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLM model.")

    parser.add_argument("--fun_nums", type=int, nargs='+', default=[10, 20, 30, 40, 50], help="Target function numbers.")
    parser.add_argument("--cand_limit", type=int, default=60, help="Max candidates to consider from retriever.")
    
    parser.add_argument("--gpu_id", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--gpu_util", type=float, default=0.9, help="GPU memory utilization.")
    parser.add_argument("--max_model_len", type=int, default=4096, help="Max model sequence length.")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    llm = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_util, 
    )
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.9,
        max_tokens=1,
    )

    subset_list = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])

    for subset in subset_list:
        try:
            process_subset(
                subset_name=subset,
                data_dir=args.data_dir,
                target_base_dir=args.target_dir,
                ori_data_base_dir=args.ori_data_dir,
                llm=llm,
                sampling_params=sampling_params,
                tokenizer=tokenizer,
                target_nums=args.fun_nums,
                cand_limit=args.cand_limit
            )
        except Exception as e:
            print(f"Error processing subset {subset}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()