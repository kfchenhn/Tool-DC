
python 1. acebench_get_data_for_ret.py \
  --input_dir "acebench_data_all_ori/data_en" \
  --ans_dir "acebench_data_all_ori/data_en/possible_answer" \
  --output_dir "acebench_data_for_retrieve/data_en"

python 1_bfcl_get_data_for_ret.py \
    --data_calling_dir "acebench_data_all_ori/data_en" \
    --data_retrieve_dir "acebench_data_for_retrieve/data_en" \
    --possible_answer_dir "acebench_data_all_ori/data_en/possible_answer"
python 2_acebench_eval_retriever.py \
    --data_root "acebench_data_for_retrieve/data_en" \
    --model_path "/path/to/your/model" \
    --retriever_type "ToolRetriever" \
    --gpu_id "0" \
    --top_k 50    

python 2_bfcl_eval_retriever.py \
    --data_root "bfcl_data_for_retrieve" \
    --retriever_type "ToolRetriever" \
    --gpu_id "1" \
    --top_k 50

python 3_acebench_get_data_call_after_retrieval.py \
    --model_path "" \
    --gpu_id "0,1" \
    --tp_size 2 \
    --data_dir "acebench_data_for_retrieve/data_en" \
    --ori_all_dir "acebench_data_all_ori/data_en" \
    --target_dir "acebench_data_extended/data_en" \
    --target_fun_num 20 \
    --max_candidates 30

python 3_bfcl_get_data_call_after_retrieval_speed.py \
    --data_dir "acebench_data_for_retrieve" \
    --target_dir "acebench_expanded" \
    --ori_data_dir "acebench_data_standard" \
    --model_path "/path/to/qwen2.5-32b" \
    --gpu_id "1" \
    --tp_size 1 \
    --fun_nums 20 50