exp_dir=./
model_base=./sft_model

for MODEL_NAME in Qwen2.5-1.5B-nothinking Qwen2.5-3B-nothinking Qwen2.5-7B-nothinking Qwen3-4B-nothinking llama3-3B-nothinking; do

    TEST_CATEGORIES="normal_single_turn_single_function,normal_single_turn_parallel_function,normal_multi_turn_user_adjust,normal_multi_turn_user_switch,normal_similar_api,normal_preference,normal_atom_bool,normal_atom_enum,normal_atom_number,normal_atom_list,normal_atom_object_deep,normal_atom_object_short" 
    
    for exp_tag in data_standard; do
        
        python infer_for_sft_model.py --exp_tag $exp_tag --model_name $MODEL_NAME --exp_dir $exp_dir --model_base $model_base --tp_size 1
        
        MODEL_RENAME=${MODEL_NAME}-local

        cd ..
        mkdir  -p ./result_all
        cp -r   $exp_dir/result_sft_all_${exp_tag}/* ./result_all/
        python eval_main.py --model $MODEL_RENAME --category normal --language en

        mkdir -p $exp_dir/score_sft_all_${exp_tag}_$MODEL_RENAME
        cp -r ./score_all $exp_dir/score_sft_all_${exp_tag}_$MODEL_RENAME/

        rm -rf ./score_all
        rm -rf ./result_all
        cd $exp_dir
    done
done