bfcl_eval_dir=../bfcl_eval
exp_dir=./
TEST_CATEGORIES="multiple,parallel_multiple,parallel,simple_java,simple_javascript,simple_python,live_multiple,live_parallel,live_simple,live_parallel_multiple"

for model_name in Llama-3.2-3B-Reflection; do


    MODEL_PATH=./sft_model/${model_name}
    for exp_tag in data_standard; do
       
        bfcl generate \
            --model meta-llama/$model_name \
            --test-category $TEST_CATEGORIES \
            --backend vllm \
            --num-gpus 1 \
            --temperature 0 \
            --gpu-memory-utilization 0.6 \
            --local-model-path $MODEL_PATH \
            --result-dir $exp_dir/result_sft_${exp_tag}

        bfcl evaluate --model meta-llama/Llama-3.2-3B-Reflection --test-category $TEST_CATEGORIES  --result-dir $exp_dir/result_sft_${exp_tag} --score-dir $exp_dir/score_sft_${exp_tag}
        
    done
done