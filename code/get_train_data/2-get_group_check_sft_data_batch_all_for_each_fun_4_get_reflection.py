import json
import os
import argparse
import ast
from string import Template
from typing import List, Tuple, Iterable, Optional

from vllm import LLM, SamplingParams 
from transformers import AutoTokenizer
from tqdm import tqdm

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
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(value, ast.NameConstant):
        output = value.value
    elif isinstance(value, ast.BinOp):
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
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
            output = {func_name: {}}
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output

def ast_parse(input_str, language="Python"):
    if language == "Python":
        parsed = ast.parse(input_str, mode="eval")
        extracted = []
        for elem in parsed.body.elts:
            assert isinstance(elem, ast.Call)
            extracted.append(resolve_ast_by_type(elem))
        return extracted
    else:
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

def tool_call_equal(pred_content, gt_tool_call):
    pred_acc = False
    if "<tool_call>" in pred_content and "</tool_call>" in pred_content:
        pred_tool_call = pred_content.split(f"<tool_call>")[-1].split(f"</tool_call>")[0]
        pred_parsed = parse_calls(pred_tool_call)
        gt_parsed = parse_calls(gt_tool_call)
        
        gt_parsed_dict = {fun_call["func"]: fun_call["kwargs"] for fun_call in gt_parsed}
        gt_fun_names = list(gt_parsed_dict.keys())

        pred_parsed_dict = {fun_call["func"]: fun_call["kwargs"] for fun_call in pred_parsed}
        pred_fun_names = list(pred_parsed_dict.keys())

        if set(pred_fun_names) == set(gt_fun_names) and all([pred_parsed_dict[name] == gt_parsed_dict[name] for name in pred_fun_names]):
            pred_acc = True
    return pred_acc

system_w_reflection_template_1 = Template("""You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out. You should only return the function call in tools call sections. Continue to output functions to call until you have fulfilled the user's request to the best of your ability.
 
Here is a list of functions in JSON format that you can invoke:

$all_fun
                                
For each interaction, you MUST strictly follow this two-step process:

**Step 1: Reasoning (<think>)**
You must engage in a detailed chain-of-thought enclosed within <think></think> tags. This process must follow these exact 3 sub-steps:
1. **Candidate Selection:** Analyze the user's question. Iterate through the function list, attempting to map extracted information to parameters. List all functions that are potential candidates.
2. **Validation:** Strictly check the candidate list against function definitions. Verify parameter types, required fields, and format constraints. Filter out any drafts that fail validation to form an effective candidate list.
3. **Final Review:** Focus exclusively on the effective candidate list (ignoring irrelevant functions). Double-check for errors such as redundant parameters, missing required values, or incorrect value types to ensure accuracy and completeness.

**Step 2: Invoke (<tool_call>)**
If you decide to invoke function(s), output them in the following specific format:
<tool_call>[func_name1(params_name1=value1, ...), func_name2(params)]</tool_call>""")

system_w_reflection_template_self_reflection_filter = Template("""You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out. You should only return the function call in tools call sections. Continue to output functions to call until you have fulfilled the user's request to the best of your ability.
 
Here is a list of functions in JSON format that you can invoke:

$all_fun                                  
For each interaction, you MUST strictly follow this two-step process:
                                                                    
**Step 1: Reasoning (<reflection>)**
Think about the reasoning process in the mind and enclosed your reasoning within  <reflection>...</reflection> XML tags.

**Step 2: Invoke (<tool_call>)**
If you decide to invoke function(s), output them in the following specific format:
<tool_call>[func_name1(params_name1=value1, ...), func_name2(params)]</tool_call>""")

answer_w_think_template = Template("""
<think>
1. **Candidate Selection:** Analyzing the user's query, I will attempt to map key information to the function parameters. The functions that potentially match and may have their parameters filled are: $cand_call
2. **Validation:** I will now strictly verify these candidates against their definitions, ensuring all parameter types and constraints are met. The functions that pass this strict verification are: $valid_call. 
3. **Final Review:** I will now eliminate any interference from irrelevant functions and focus solely on the valid candidates. $reflection
</think>
<tool_call>$tool_call</tool_call>
""")

def main():
    parser = argparse.ArgumentParser(description="Run reflection filtering and data processing.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to N1_tool-w_individual_call.json")
    parser.add_argument("--output_correct_path", type=str, required=True, help="Path to save correct reflection data")
    parser.add_argument("--output_not_correct_path", type=str, required=True, help="Path to save incorrect reflection data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--gpu_ids", type=str, default="2,3", help="CUDA_VISIBLE_DEVICES string (e.g., '0,1')")
    parser.add_argument("--tp_size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--max_len", type=int, default=16384, help="Maximum model length")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    with open(args.input_path, "r") as f:
        datalist_w_individual_call = json.load(f)

    print(f"##### len(datalist_w_individual_call): {len(datalist_w_individual_call)} #######")
    
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

    all_prompts_2 = []
    for dataitem in tqdm(datalist_w_individual_call, desc="Preparing Prompts"):
        system = dataitem["system"]
        question = dataitem["query"]
        fun_list_str = system.split("Here is a list of functions in json format that you can invoke.")[-1].strip()
        fun_list = json.loads(fun_list_str)
        valid_call = dataitem["valid_call"]
        valid_fun_list = [fun for fun in fun_list if fun["name"] in valid_call]

        system_message = system_w_reflection_template_self_reflection_filter.substitute(
            all_fun=json.dumps(valid_fun_list, ensure_ascii=False, indent=4),
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]

        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompts_2.append(prompt)

    outputs = llm.generate(all_prompts_2, sampling_params, use_tqdm=True)

    correct_datalist = []
    not_correct_datalist = []
    
    for out_id, out in enumerate(outputs):
        pred_content = out.outputs[0].text + "</tool_call>"
        dataitem = datalist_w_individual_call[out_id]

        system = dataitem["system"]
        question = dataitem["query"]
        cand_call = dataitem["cand_call"]
        valid_call = dataitem["valid_call"]
        fun_list_str = system.split("Here is a list of functions in json format that you can invoke.")[-1].strip()
        fun_list = json.loads(fun_list_str)
        gt_tool_call = dataitem["answer"]

        is_tool_equal = tool_call_equal(pred_content, gt_tool_call)
        has_reflection = "<reflection>" in pred_content and "</reflection>" in pred_content
        has_tool_call = "<tool_call>" in pred_content and "</tool_call>" in pred_content

        if has_tool_call and has_reflection and is_tool_equal:
            system_w_reflection = system_w_reflection_template_1.substitute(
                all_fun=json.dumps(fun_list, ensure_ascii=False, indent=4),
            )
            reflection_content = pred_content.split("<reflection>")[1].split("</reflection>")[0]
            answer_w_reflection = answer_w_think_template.substitute(
                cand_call=cand_call, 
                valid_call=valid_call, 
                reflection=reflection_content, 
                tool_call=gt_tool_call
            )
            correct_datalist.append({
                "system": system,
                "query": question,
                "answer": gt_tool_call,
                "system_w_reflection": system_w_reflection,
                "answer_w_reflection": answer_w_reflection
            })
        else:
            system_w_reflection = system_w_reflection_template_1.substitute(
                all_fun=json.dumps(fun_list, ensure_ascii=False, indent=4),
            )
            not_correct_datalist.append({
                "system": system,
                "query": question,
                "answer": gt_tool_call,
                "system_w_reflection": system_w_reflection,
                "answer_w_reflection": pred_content
            })

    print(f"##### len(correct_datalist): {len(correct_datalist)} #######")
    
    with open(args.output_correct_path, "w") as f:
        json.dump(correct_datalist, f, indent=4, ensure_ascii=False)
    with open(args.output_not_correct_path, "w") as f:
        json.dump(not_correct_datalist, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()