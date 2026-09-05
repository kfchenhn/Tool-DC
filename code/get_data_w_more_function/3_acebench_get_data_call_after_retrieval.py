import os
import json
import csv
import copy
import random
import argparse
from typing import List, Dict
from string import Template

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

def read_tsv_as_dicts(path: str, encoding: str = "utf-8") -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]

def get_args():
    parser = argparse.ArgumentParser(description="Expand functions using vLLM for deduplication")
    parser.add_argument("--model_path", type=str, required=True, help="Path to vLLM model")
    parser.add_argument("--gpu_id", type=str, default="0,1", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--tp_size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--gpu_util", type=float, default=0.9, help="GPU memory utilization")
    
    parser.add_argument("--data_dir", type=str, required=True, help="Path to acebench_data_for_retrieve/data_en")
    parser.add_argument("--ori_all_dir", type=str, required=True, help="Path to acebench_data_all_ori/data_en")
    parser.add_argument("--target_dir", type=str, required=True, help="Where to save the expanded .json files")
    
    parser.add_argument("--target_fun_num", type=int, default=20, help="Stop adding functions when reached this number")
    parser.add_argument("--max_candidates", type=int, default=30, help="Max candidates to check from retriever per query")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    
    return parser.parse_args()

def main():
    args = get_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.makedirs(args.target_dir, exist_ok=True)

    
    llm = LLM(
        model=args.model_path,
        max_model_len=4096,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_util,
        trust_remote_code=True
    )
    
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.9,
        max_tokens=1,
    )
    
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    for subset in os.listdir(args.data_dir):
        subdir = os.path.join(args.data_dir, subset)
        if not os.path.isdir(subdir):
            continue

   
        retr_corpus_id_path = os.path.join(subdir, "ori/ToolRetriever/test_retr_corpus_id.json")
        corpus_path = os.path.join(subdir, "ori/corpus.tsv")
        ori_data_path = os.path.join(args.ori_all_dir, subset + ".json")
        
        if not (os.path.exists(retr_corpus_id_path) and os.path.exists(corpus_path) and os.path.exists(ori_data_path)):
           
            continue

        corpus_list = read_tsv_as_dicts(corpus_path)
        corpus_dict = {row["docid"]: row["document_content"] for row in corpus_list}

        ori_datalist = []
        with open(ori_data_path, "r", encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    ori_datalist.append(json.loads(line))

        with open(retr_corpus_id_path, "r", encoding='utf-8') as f:
            retr_corpus_id_list = json.load(f)
        
        new_datalist = []

        for idx, data_item in enumerate(ori_datalist):
            ori_fun_list = data_item["function"] 
            
            if len(ori_fun_list) >= args.target_fun_num:
                new_datalist.append(data_item)
                continue

            corpus_id_list = retr_corpus_id_list[idx]["corpus_id_list"]
            candidate_fun_list = []
            for doc_id in corpus_id_list:
                fun_str = corpus_dict.get(str(doc_id))
                if fun_str:
                    candidate_fun_list.append(json.loads(fun_str))

            candidate_fun_list = candidate_fun_list[:args.max_candidates]
            
            batch_prompts = []
            prompt_candidate_map = [] 
            origin_fun_names = {f["name"] for f in ori_fun_list}

            for cand_fun in candidate_fun_list:
                if cand_fun["name"] in origin_fun_names:
                    continue

                for exist_fun in ori_fun_list:
                    prompt_candidate_map.append(cand_fun)
                    user_message = PROMPT_TEMPLATE.substitute(
                        function1=json.dumps(cand_fun, indent=4),
                        function2=json.dumps(exist_fun, indent=4),
                    )
                    
                    messages = [{"role": "user", "content": user_message}]
                    text_input = tok.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    batch_prompts.append(text_input)

            blacklisted_names = set() 
            if batch_prompts:
                outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
                for i, out in enumerate(outputs):
                    ans = out.outputs[0].text.strip()
                    current_cand_fun = prompt_candidate_map[i]
                    if ans in ["1", "2"]:
                        blacklisted_names.add(current_cand_fun["name"])

            final_fun_names = set(origin_fun_names) 
            for cand_fun in candidate_fun_list:
                if len(data_item["function"]) >= args.target_fun_num:
                    break
                
                name = cand_fun["name"]
                if (name not in final_fun_names) and (name not in blacklisted_names):
                    data_item["function"].append(cand_fun)
                    final_fun_names.add(name)

            random.Random(args.seed).shuffle(data_item["function"])
            new_datalist.append(data_item)

        output_file = os.path.join(args.target_dir, subset + ".json")
        with open(output_file, "w", encoding='utf-8') as f:
            for obj in new_datalist:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        
     
if __name__ == "__main__":
    main()