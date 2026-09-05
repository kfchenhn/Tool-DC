

MODEL_DIR=models
exp_dir=./
TEST_CATEGORIES="multiple,parallel_multiple,parallel,simple_java,simple_javascript,simple_python,live_multiple,live_parallel,live_simple,live_parallel_multiple"



for exp_tag in data_gt_funs data_standard data_extended; do

    cp -r $exp_dir/$exp_tag/* ../bfcl_eval/data/
        for MODEL_NAME in Qwen2.5-3B-Instruct; do
        MODEL_PATH=$MODEL_DIR/$MODEL_NAME
        bash bfcl_test_and_eval_qwen.sh $MODEL_NAME $TEST_CATEGORIES $MODEL_PATH $exp_dir $exp_tag
    done

    cp -r $exp_dir/data_standard/* ../bfcl_eval/data/
done


for NUM_GROUP in 5; do
for MODEL_NAME in Qwen2.5-1.5B-Instruct Qwen2.5-14B-Instruct Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct; do
    for GROUP_METHOD in sad swod sdecouple; do 
        for exp_tag in data_extended data_standard; do

            cp -r $exp_dir/$exp_tag/* ../bfcl_eval/data/
            TEST_CATEGORIES="multiple,parallel_multiple,parallel,simple_java,simple_javascript,simple_python,live_multiple,live_parallel,live_simple,live_parallel_multiple"

            MODEL_PATH=$MODEL_DIR/$MODEL_NAME
       
            python bfcl_infer_w_group_similarity_and_decouple.py --model_name $MODEL_NAME --exp_dir $exp_dir --exp_tag $exp_tag --model_path $MODEL_PATH --result_dir $exp_dir/result_reflect_sad_$exp_tag --data_categories $TEST_CATEGORIES --num_groups ${NUM_GROUP} --group_method $GROUP_METHOD --tensor_parallel_size 1

            bfcl evaluate --model Qwen/$MODEL_NAME --test-category $TEST_CATEGORIES  --result-dir $exp_dir/result_reflect_sad_${exp_tag}_${GROUP_METHOD}_${NUM_GROUP} --score-dir $exp_dir/score_reflect_sad_${exp_tag}_${GROUP_METHOD}_${NUM_GROUP}



            cp -r $exp_dir/data_standard/* ../bfcl_eval/data/
        done
    done
done

done