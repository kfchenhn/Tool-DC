import json
import os
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from string import Template


SYSTEM_PROMPT_FOR_NORMAL_DATA_EN = """You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out. You should only return the function call in tools call sections. Continue to output functions to call until you have fulfilled the user's request to the best of your ability.

{time}

Here is a list of functions in JSON format that you can invoke:

{function}
    
                              
For each interaction, you MUST strictly follow this two-step process:

**Step 1: Reasoning (<think>)**
You must engage in a detailed chain-of-thought enclosed within <think></think> tags. This process must follow these exact 3 sub-steps:
1. **Candidate Selection:** Analyze the user's question. Iterate through the function list, attempting to map extracted information to parameters. List all functions that are potential candidates.
2. **Validation:** Strictly check the candidate list against function definitions. Verify parameter types, required fields, and format constraints. Filter out any drafts that fail validation to form an effective candidate list.
3. **Final Review:** Focus exclusively on the effective candidate list (ignoring irrelevant functions). Double-check for errors such as redundant parameters, missing required values, or incorrect value types to ensure accuracy and completeness.

**Step 2: Invoke (<tool_call>)**
If you decide to invoke function(s), output them in the following specific format:
<tool_call>[func_name1(params_name1=value1, ...), func_name2(params)]</tool_call>"""



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
    model_path = os.path.join(args.model_base, args.model_name)

    data_category_list = [
        "normal_single_turn_single_function", 
        "normal_single_turn_parallel_function",
        "normal_multi_turn_user_adjust",
        "normal_multi_turn_user_switch",
        "normal_similar_api",
        "normal_preference",
        "normal_atom_bool",
        "normal_atom_enum",
        "normal_atom_number",
        "normal_atom_list",
        "normal_atom_object_deep",
        "normal_atom_object_short"
    ]


    llm = LLM(
        model=model_path,
        max_model_len=16384,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_mem_util,
    )
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.9,
        max_tokens=512,
        logprobs=2,
    )
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    for data_category in data_category_list:
        data_path = f"{args.exp_dir}/{args.exp_tag}/data_en/data_{data_category}.json"
        result_path = f"{args.exp_dir}/result_sft_all_{args.exp_tag}/result_en/{args.model_name}-local/data_{data_category}_result.json"

        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        datalist = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                datalist.append(json.loads(line))

        result_list = []

        all_prompt = []
        for data_item in tqdm(datalist):
            id = data_item["id"]
            fun_list = data_item["function"]
            question = data_item["question"]
            now_time = data_item.get("time", "") if len(data_item.get("time", "")) > 10 else ""
            my_profile = data_item.get("profile", "")

            system_message = SYSTEM_PROMPT_FOR_NORMAL_DATA_EN.format(
                time=now_time,
                function=fun_list
            )
       
            user_message = question
     
            if "preference" in id:
                user_message = preference_user_template.substitute(profile=my_profile, question=question)

            if "multi_turn" in id or "similar" in id:
                user_message = multi_turn_user_template.substitute(question=question)


            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_prompt.append(prompt)
        outputs = llm.generate(all_prompt, sampling_params, use_tqdm=False)
        for idx, out in enumerate(outputs):

            raw_answer = out.outputs[0].text
            data_item = datalist[idx]
            
            if "<tool_call>" in raw_answer and "</tool_call>" in raw_answer:
                    raw_answer  = raw_answer.split("<tool_call>")[1].split("</tool_call>")[0]
                    raw_answer = raw_answer.strip()
            
            result_list.append(
                {"id": id, "result": raw_answer}
            )

        with open(result_path, "w", encoding="utf-8") as f:
            for res in result_list:
                f.write(json.dumps(res, ensure_ascii=False, separators=(',', ':')) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_tag", type=str, required=True, help="data mode, e.g., ori")
    parser.add_argument("--model_name", type=str, required=True, help="Model name, e.g., Qwen3-8B")
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory")
    parser.add_argument("--model_base", type=str, required=True, help="Base path of models")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu_mem_util", type=float, default=0.8, help="GPU memory utilization")
    args = parser.parse_args()
    main(args)
