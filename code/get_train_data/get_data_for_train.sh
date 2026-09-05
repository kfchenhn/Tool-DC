python 1-get_group_check_sft_data_batch_all_for_each_fun_4_get_individue.py \
    --model_path models/Qwen2.5-7B-Instruct/ \
    --input_path /xalm_tool.json \
    --output_path /xalm_tool-w_individual_call.json \
    --gpu_ids 4,5,6,7 \
    --tp_size 4

python 2-get_group_check_sft_data_batch_all_for_each_fun_4_get_reflection.py \
    --input_path "/xalm_tool-w_individual_call.json" \
    --output_correct_path "/xalm_tool-reflection_correct_1216.json" \
    --output_not_correct_path "/xalm_tool-reflection_not_correct_1216.json" \
    --model_path "" \
    --gpu_ids "2,3" \
    --tp_size 2