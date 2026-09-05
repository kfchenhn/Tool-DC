import json
import os
import argparse
import ast
import numpy as np
from string import Template
from typing import List, Tuple, Iterable, Optional

from vllm import LLM, SamplingParams 
from transformers import AutoTokenizer
from tqdm import tqdm

def get_reflection(pred_1, answer, fun_list):
    reflection_list = {
        "redundant_parameter": [],
        "missing_parameter": [],
        "incorrect_value": [],
        "incorrect_datatype": [],
    }
    pred_parsed = parse_calls(pred_1)        
    gt_parsed = parse_calls(answer)
    
    pred_parsed_dict = {}
    gt_parsed_dict = {}
    all_fun_dict = {}

    for fun_call in pred_parsed:
        pred_parsed_dict[fun_call["func"]] = fun_call["kwargs"]
    for fun_call in gt_parsed:
        gt_parsed_dict[fun_call["func"]] = fun_call["kwargs"]
    for fun in fun_list:
        all_fun_dict[fun["name"]] =  fun["parameters"]["properties"]
    
    pred_fun_names = list(pred_parsed_dict.keys())
    gt_fun_names = list(gt_parsed_dict.keys())
        
    redundant_fun = list(set(pred_fun_names) - set(gt_fun_names))
    missing_fun = list(set(gt_fun_names) - set(pred_fun_names))

    fun_reflection_names = list(set(pred_fun_names) & set(gt_fun_names))    
    if len(fun_reflection_names) > 0:
        for fun_name in fun_reflection_names:
            params_in_pred = pred_parsed_dict[fun_name]
            params_in_gt = gt_parsed_dict[fun_name]
            params_in_docs = all_fun_dict[fun_name]
            
            params_in_pred_name_list = list(params_in_pred.keys())
            params_in_gt_name_list = list(params_in_gt.keys())

            redundant_params = list(set(params_in_pred_name_list) - set(params_in_gt_name_list))
            missing_params = list(set(params_in_gt_name_list) - set(params_in_pred_name_list))
            
            reflection_list["redundant_parameter"].extend(f"{fun_name}.{p_name}" for p_name in redundant_params)
            reflection_list["missing_parameter"].extend(f"{fun_name}.{p_name}" for p_name in missing_params)

            params_reflection_names = list(set(params_in_pred_name_list) & set(params_in_gt_name_list))
            
            for p_name in params_reflection_names:
                args_type_all_right = True
                value = params_in_pred[p_name]
                
                if isinstance(params_in_docs[p_name]["type"], list):
                    continue
                target_type = params_in_docs[p_name]["type"].lower()
                target_value = params_in_gt[p_name]
                
                if target_type in ["integer", "int", "float"]:
                    if not value == target_value:
                       reflection_list["incorrect_value"].append(f"{fun_name}.{p_name}")

                if target_type == "string":
                    if not isinstance(value, str):
                        args_type_all_right = False
                elif target_type in ["integer", "int"]:
                    if not isinstance(value, int):
                        args_type_all_right = False
                elif target_type == "float":
                    if not isinstance(value, (float, int)):
                        args_type_all_right = False
                elif target_type in ["boolean", "bool"]:
                    if not (str(value).lower() == "true" or str(value).lower() == "false"):
                        args_type_all_right = False
                elif target_type == "dict":
                    if not isinstance(value, dict):
                        args_type_all_right = False
                elif target_type == "array":
                    if not isinstance(value, list):
                        args_type_all_right = False

                if not args_type_all_right:
                    reflection_list["incorrect_datatype"].append(f"{fun_name}.{p_name}")

    funs_hit = fun_reflection_names
    return funs_hit, reflection_list

def resolve_ast_call(elem):
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}

def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        return "..." if value.value is Ellipsis else value.value
    elif isinstance(value, ast.UnaryOp):
        return -value.operand.value
    elif isinstance(value, ast.List):
        return [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        return {resolve_ast_by_type(k): resolve_ast_by_type(v) for k, v in zip(value.keys, value.values)}
    elif isinstance(value, ast.NameConstant):
        return value.value
    elif isinstance(value, ast.BinOp):
        return eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        return value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            func_parts = []
            func_part = value.func
            while isinstance(func_part, ast.Attribute):
                func_parts.append(func_part.attr)
                func_part = func_part.value
            if isinstance(func_part, ast.Name):
                func_parts.append(func_part.id)
            func_name = ".".join(reversed(func_parts))
            return {func_name: {}}
        return resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        return tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        return eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        return "..."
    elif isinstance(value, ast.Subscript):
        try:
            return ast.unparse(value.body[0].value)
        except:
            return ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")

def ast_parse(input_str, language="Python"):
    if language == "Python":
        parsed = ast.parse(input_str, mode="eval")
        extracted = []
        for elem in parsed.body.elts:
            assert isinstance(elem, ast.Call)
            extracted.append(resolve_ast_by_type(elem))
        return extracted
    raise NotImplementedError(f"Unsupported language: {language}")

def parse_calls(s: str):
    try:
        calls = ast_parse(s)
    except Exception:
        return []
    results = []
    for c in calls:
        try:
            func = list(c.keys())[0]
            kwargs = list(c.values())[0]
            results.append({"func": func, "kwargs": kwargs})
        except Exception:
            continue
    return results

def normalize_booleans_and_null(text: str) -> str:
    return text.replace("=true", "=True").replace("=false", "=False").replace("null", "None").strip()

pre_sys_template = Template("""You are an expert in composing functions. You are given a question and a function.
For the function, attempt to map the user's question to it. Output:
[func_name(param1=value1, param2=value2, ...)]
You SHOULD NOT include any other text in the response.
If required parameters are missing, do **not** invent values. Instead, use the function name with empty parentheses.

Here is the function in JSON format that you can invoke:
$all_fun
""")

def main():
    parser = argparse.ArgumentParser(description="Run VLLM Tool-CG reflection process.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input N1_tool.json.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save individual call results.")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="CUDA_VISIBLE_DEVICES string.")
    parser.add_argument("--tp_size", type=int, default=4, help="Tensor parallel size.")
    parser.add_argument("--max_len", type=int, default=16384, help="Max model length.")
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    llm = LLM(
        model=args.model_path,
        max_model_len=args.max_len,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=0.75,
    )
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.9,
        max_tokens=2048,
    )
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    with open(args.input_path, "r") as f:
        datalist_raw = json.load(f)
    
    datalist = []
    for dataitem in datalist_raw:
        system = dataitem["system"]
        fun_list_str = system.split("Here is a list of functions in json format that you can invoke.")[-1].strip()
        try:
            json.loads(fun_list_str)
            datalist.append(dataitem)
        except:
            continue

    refine_datalist = datalist
    print(f"Total items to process: {len(refine_datalist)}")

    all_prompts_1 = []
    metadata_map = [] 

    for di, dataitem in tqdm(enumerate(refine_datalist), desc="Building Prompts"):
        system = dataitem["system"]
        question = dataitem["query"]
        fun_list_str = system.split("Here is a list of functions in json format that you can invoke.")[-1].strip()
        fun_list = json.loads(fun_list_str)
        
        groups = [[i] for i in range(len(fun_list))]

        for gi, grp in enumerate(groups):
            cand_funcs = [fun_list[i] for i in grp]
            system_message = pre_sys_template.substitute(
                all_fun=json.dumps(cand_funcs, ensure_ascii=False, indent=4),
            )        
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question}
            ]

            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_prompts_1.append(prompt)
            metadata_map.append((di, gi))

    outputs = llm.generate(all_prompts_1, sampling_params, use_tqdm=True)

    results_by_dataitem = [{} for _ in refine_datalist]
    for out_idx, out in enumerate(outputs):
        di, gi = metadata_map[out_idx]
        results_by_dataitem[di][gi] = normalize_booleans_and_null(out.outputs[0].text.strip())

    datalist_w_individual_call = []
    for di, dataitem in tqdm(enumerate(refine_datalist), desc="Post-processing"):
        try:
            system = dataitem["system"]
            question = dataitem["query"]
            gt_answer = dataitem["answer"]
            
            fun_list_str = system.split("Here is a list of functions in json format that you can invoke.")[-1].strip()
            fun_list = json.loads(fun_list_str)
            
            local_outputs = results_by_dataitem[di]
            pred_tool_call_list = list(local_outputs.values())

            cand_call = "["
            valid_call = "["
            
            for fun_call in pred_tool_call_list:
                if not fun_call.startswith("["): fun_call = "[" + fun_call
                if not fun_call.endswith("]"): fun_call = fun_call + "]"
                
                fun_hit, _ = get_reflection(fun_call, gt_answer, fun_list)
                
                if "=" in fun_call:
                    cand_call += fun_call[1:-1] + ", "
                if len(fun_hit) > 0:
                    for fun_name in fun_hit:
                        valid_call += fun_name + ", "
            
            cand_call = (cand_call[:-2] if len(cand_call) > 1 else "[") + "]"
            valid_call = (valid_call[:-2] if len(valid_call) > 1 else "[") + "]"

            datalist_w_individual_call.append({
                "system": system,
                "query": question,
                "answer": gt_answer,
                "cand_call": cand_call,
                "valid_call": valid_call
            })
        except Exception as e:
            continue

    print(f"Final dataset size: {len(datalist_w_individual_call)}")
    with open(args.output_path, "w") as f:
        json.dump(datalist_w_individual_call, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()