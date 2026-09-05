
MODEL_NAME=$1
TEST_CATEGORIES=$2
MODEL_PATH=$3
exp_dir=$4
exp_tag=$5

bfcl generate \
    --model Qwen/$MODEL_NAME \
    --test-category $TEST_CATEGORIES \
    --backend vllm \
    --num-gpus 1 \
    --temperature 0 \
    --gpu-memory-utilization 0.6 \
    --local-model-path $MODEL_PATH \
    --result-dir $exp_dir/result_$exp_tag

bfcl evaluate --model Qwen/$MODEL_NAME --test-category $TEST_CATEGORIES  --result-dir $exp_dir/result_$exp_tag --score-dir $exp_dir/score_$exp_tag
