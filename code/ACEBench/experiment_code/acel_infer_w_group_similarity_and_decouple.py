import json
import os
import re
import ast
import sys
import math
import random
import argparse
from string import Template
from typing import List, Tuple, Iterable, Optional

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from rank_bm25 import BM25Okapi

def _balanced_chunks(indexes: List[int], num_groups: int) -> List[Tuple[int, ...]]:
    n = len(indexes)
    if n == 0:
        return []
    num_groups = min(num_groups, n)
    base, rem = divmod(n, num_groups)
    out, start = [], 0
    for g in range(num_groups):
        size = base + (1 if g < rem else 0)
        out.append(tuple(indexes[start:start+size]))
        start += size
    return out

def split_round_robin(g, num_groups):
    buckets = [[] for _ in range(num_groups)]
    for idx, item in enumerate(g):
        buckets[idx % num_groups].append(item)
    return buckets

def split_and_regroup(groups, target_num_groups):
    num_groups = target_num_groups
    result = [[] for _ in range(num_groups)]

    for g in groups:
        splits = split_round_robin(g, num_groups)
        for i in range(num_groups):
            result[i].extend(splits[i])
    return result

def _group_similarity_bm25(question: str, fun_list: List[dict], num_groups: int) -> List[Tuple[int, ...]]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return _balanced_chunks(list(range(len(fun_list))), num_groups)

    docs = []
    for f in fun_list:
        if isinstance(f, dict) and 'description' in f and isinstance(f['description'], str):
            docs.append(f['description'])
        else:
            docs.append(str(f))

    tokenized_docs = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)
    sorted_idx = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]

    return _balanced_chunks(sorted_idx, num_groups)

def chunk_indices_by_groups(
    question: str,
    fun_list: List[dict],
    num_groups: int,
    group_method: str = "random",
    random_seed: int = 42,
) -> List[Tuple[int, ...]]:
    n = len(fun_list)
    if n == 0: return []
    num_groups = min(num_groups, n)
    similarity_groups_tuples = _group_similarity_bm25(question, fun_list, len(fun_list)//num_groups) 
    similarity_groups_list = [list(g) for g in similarity_groups_tuples]
    result_lists = split_and_regroup(similarity_groups_list, target_num_groups=num_groups)
    result_lists.append(similarity_groups_list[0])
    result_lists = result_lists[::-1]
    return [tuple(v) for v in result_lists]


def parse_calls(s: str):
    try:
        tree = ast.parse(s, mode="eval").body
        calls = tree.elts if isinstance(tree, ast.List) else [tree]
    except Exception:
        return None

    results = []
    for c in calls:
        if not isinstance(c, ast.Call): continue
        try:
            func = ".".join(ast.unparse(c.func).split())
            args = [ast.literal_eval(a) for a in c.args]
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in c.keywords if kw.arg}
            results.append({"func": func, "args": args, "kwargs": kwargs})
        except Exception:
            continue
    return results

def normalize_booleans_and_null(text: str) -> str:
    return text.replace("=true", "=True").replace("=false", "=False").replace("null", "None").strip()

def build_message_from_question(question, base_system):
    user_message = question
    system_message = base_system
    if isinstance(question, list): 
        conv = question[0]
        if isinstance(conv, list):
             for c in conv:
                if c.get("role") == "system": system_message += c.get("content", "")
                if c.get("role") == "user": user_message = c.get("content", "")
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    return messages, user_message, system_message

def extract_func_names_only(s: str) -> List[str]:
    if not s or s.strip() == "[]":
        return []
    pattern = r'([\w\.]+)(?:\s*\(|\b)'
    found = re.findall(pattern, s.replace("[", "").replace("]", "").replace(",", " "))
    return [f.strip() for f in found if f.strip()]

pre_sys_template = Template(
"""You are a Function Selection Expert. Your task is to identify ALL functions that are semantically relevant to the user's question from the provided list. Extract information from the user's question and substitute it into the function parameters.

Read the user's question and the function descriptions carefully. Choose any function that *could* potentially meet user needs or meet a part of user needs. 
If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)].  
You SHOULD NOT include any other text in the response.

Here is a list of functions in json format that you can invoke.
$all_fun

$time
"""
)

refine_sys_template = Template(
"""You are an expert in composing functions.You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)].  You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in json format that you can invoke.
$all_fun

$time
"""
)

preference_user_template = Template(
"""You are given a character profile and a question:

Character Profile:
$profile

Question:
$question
"""
)

multi_turn_user_template = Template(
"""
Here is a question with some conversation history:

$question

"""
)

def main(args):
    model_name = args.model_name
    exp_dir = args.exp_dir
    llm = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    data_category_list = args.data_categories.split(",")

    for data_category in data_category_list:
        data_path = os.path.join(exp_dir, args.exp_tag, "data_en", f"data_{data_category}.json")
        result_path = os.path.join(args.result_dir + f"_{args.group_method}", f"result_all_{args.exp_tag}", "result_en", f"{model_name}-local-ours", f"data_{data_category}_result.json")

        if os.path.exists(result_path):
            continue
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        
        datalist = []
        if not os.path.exists(data_path):
            continue

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                datalist.append(json.loads(line))

        all_prompts_1 = []
        metadata_map = []  
        groups_list = []   
        
        for di, data_item in tqdm(enumerate(datalist)):
            id = data_item["id"]
            fun_list = data_item["function"]
            question = data_item["question"]
            now_time = ""
            if "time" in data_item:
                now_time = "When answering questions involving time, remember:\n" + data_item["time"]
            my_profile = ""
            if "profile" in data_item:
                my_profile = data_item["profile"]

            num_groups = args.num_groups
            if len(fun_list) < num_groups:
                num_groups = len(fun_list)
            
            groups = chunk_indices_by_groups(
                question=str(question), 
                fun_list=fun_list, 
                num_groups=num_groups, 
                group_method=args.group_method
            )
            groups_list.append(groups)

            for gi, grp in enumerate(groups):
                cand_funcs = [fun_list[i] for i in grp]
                system_message = pre_sys_template.substitute(
                    all_fun=json.dumps(cand_funcs,  indent=4),
                    time=now_time,
                )

                if "preference" in id:
                    question = preference_user_template.substitute(profile=my_profile, question=question)

                if "multi_turn" in id or "similar" in id:
                    question = multi_turn_user_template.substitute(question=question)

                messages, _, _ = build_message_from_question(question, system_message)
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                all_prompts_1.append(prompt)
                metadata_map.append((di, gi))

        if not all_prompts_1:
            continue
            
        outputs_1 = llm.generate(all_prompts_1, sampling_params, use_tqdm=True)

        results_by_dataitem = [{} for _ in datalist]
        for out_idx, out in enumerate(outputs_1):
            di, gi = metadata_map[out_idx]
            text = normalize_booleans_and_null(out.outputs[0].text.strip())
            if "<think>" in text and "</think>" in text:
                text = text.split("</think>\n\n")[-1]
            results_by_dataitem[di][gi] = text

        all_prompts_2 = []
        phase2_data_indices = [] 

        for di, data_item in tqdm(enumerate(datalist)):
            id = data_item["id"]
            fun_list = data_item["function"]
            fun_name_list = [fun["name"] for fun in fun_list]
            question = data_item["question"]
            now_time = ""
            if "time" in data_item:
                now_time = "When answering questions involving time, remember:\n" + data_item["time"]
            my_profile = ""
            if "profile" in data_item:
                my_profile = data_item["profile"]

            fun_required_args_list = [
                set(fun.get("arguments", {}).get("required", [])) 
                if "arguments" in fun 
                else set(fun.get("parameters", {}).get("required", [])) 
                for fun in fun_list
            ]
            
            fun_all_args_list = [
                set(list(fun["arguments"]["properties"].keys())) 
                if "arguments" in fun 
                else set(list(fun["parameters"]["properties"].keys())) 
                for fun in fun_list
            ]

            local_outputs = results_by_dataitem[di]
            groups = groups_list[di]
          
            tool_call_result_names = []
            hints = []
            for gi, raw_answer in local_outputs.items():
                grp = groups[gi]
                cand_names_in_grp = [fun_name_list[idx] for idx in grp]
                names_in_answer = extract_func_names_only(raw_answer)
                for na in names_in_answer:
                    if na in cand_names_in_grp:
                        tool_call_result_names.append(na)

                parsed = parse_calls(raw_answer)
                if parsed:    
                    for fun_call in parsed:
                        func_name = fun_call["func"]
                        if func_name not in fun_name_list:
                            continue
                        else:
                            fun_hit_idx = fun_name_list.index(func_name)
                            req_set = fun_required_args_list[fun_hit_idx]
                            all_set = fun_all_args_list[fun_hit_idx]
                            kw_keys = set(fun_call["kwargs"].keys())

                            missing_params = list(req_set - kw_keys)
                            redundant_params = list(kw_keys - all_set)
                            if len(missing_params) > 0:
                                hints.append(f"For function '{func_name}', you must provide parameters: {str(missing_params)}.")
                                continue
                            if len(redundant_params) > 0:
                                hints.append(f"For function '{func_name}', remove invalid parameters: {str(redundant_params)}.")
                                continue
            
            tool_call_result_names = list(dict.fromkeys(tool_call_result_names))
            retain_fun_list = [fun_list[fun_name_list.index(name)] for name in tool_call_result_names]

            refine_system_message = refine_sys_template.substitute(
                all_fun=json.dumps(retain_fun_list,  indent=4),
                time=now_time,
            )
            if "preference" in id:
                question = preference_user_template.substitute(profile=my_profile, question=question)

            if "multi_turn" in id or "similar" in id:
                question = multi_turn_user_template.substitute(question=question)

            if hints:
                hint_text = "\n".join(hints)
                refine_system_message += f"\n\nFeedback from previous attempt:\n{hint_text}\n\nPlease generate the correct function call based on the feedback."
            
            refine_messages, _, _ = build_message_from_question(question, refine_system_message)
            refine_prompt = tok.apply_chat_template(refine_messages, tokenize=False, add_generation_prompt=True)
            
            all_prompts_2.append(refine_prompt)
            phase2_data_indices.append(di)

        if not all_prompts_2:
            result_list = [{"id": datalist[i]["id"], "result": "[]"} for i in range(len(datalist))]
        else:
            refine_outputs = llm.generate(all_prompts_2, sampling_params, use_tqdm=True)
            
            result_map = {} 
            for i, out in enumerate(refine_outputs):
                original_di = phase2_data_indices[i]
                answer = out.outputs[0].text.strip()
                if "<think>" in answer and "</think>" in answer:
                    answer = answer.split("</think>\n\n")[-1]
                result_map[original_di] = answer
            
            result_list = []
            for di, item in enumerate(datalist):
                res_str = result_map.get(di, "[]")
                result_list.append({
                    "id": item["id"],
                    "result": res_str
                })

        with open(result_path, "w", encoding="utf-8") as f:
            for res in result_list:
                f.write(json.dumps(res,  separators=(',', ':')) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--exp_tag", type=str, default="data_all")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--data_categories", type=str, default="normal_single_turn_single_function,normal_single_turn_parallel_function")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--num_groups", type=int, default=3)
    parser.add_argument("--group_method", type=str, default="sad") 

    args = parser.parse_args()
    main(args)