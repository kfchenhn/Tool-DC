exp_dir=./

for MODEL_NAME in Llama-3.2-3B-Instruct; do
    MODEL_PATH=./models/${MODEL_NAME}
    TEST_CATEGORIES="normal_single_turn_single_function,normal_single_turn_parallel_function,normal_multi_turn_user_adjust,normal_multi_turn_user_switch,normal_similar_api,normal_preference,normal_atom_bool,normal_atom_enum,normal_atom_number,normal_atom_list,normal_atom_object_deep,normal_atom_object_short" 
    

    for exp_tag in data_gt_funs data_standard; do
        cd ..
     
        cp -r ${exp_dir}/$exp_tag/* ../data_all/
        MODEL_RENAME=$MODEL_NAME-local
        CUDA_VISIBLE_DEVICES=7 python generate.py --model $MODEL_RENAME --model-path $MODEL_PATH --category normal --language en --num-gpus 1
        mkdir -p $exp_dir/result_ori_method_${exp_tag}
        cp -r  ./result_all/* $exp_dir/result_ori_method_${exp_tag}/

        python eval_main.py --model $MODEL_RENAME --category normal --language en


        mkdir -p $exp_dir/score_ori_method_${exp_tag}_$MODEL_RENAME
        cp -r ./score_all $exp_dir/score_ori_method_${exp_tag}_$MODEL_RENAME/

        rm -rf ./score_all
        rm -rf ./result_all
        
        cp -r ${exp_dir}/data_standard/* ../data_all/
    done
done



for NUM_GROUP in 5; do
    for MODEL_NAME in $1; do
        MODEL_PATH=../models/${MODEL_NAME}
        TEST_CATEGORIES="normal_single_turn_single_function,normal_single_turn_parallel_function,normal_multi_turn_user_adjust,normal_multi_turn_user_switch,normal_similar_api,normal_preference,normal_atom_bool,normal_atom_enum,normal_atom_number,normal_atom_list,normal_atom_object_deep,normal_atom_object_short" 
        
        GROUP_METHOD=sad
        rename_model_name=${MODEL_NAME}-local-ours
        for exp_tag in data_standard; do
            python ace_infer_w_group_similarity_and_decouple.py --model_name $MODEL_NAME --exp_dir $exp_dir --exp_tag $exp_tag --model_path $MODEL_PATH --result_dir $exp_dir/result_reflect_sad_${NUM_GROUP}_$exp_tag --data_categories $TEST_CATEGORIES --num_groups ${NUM_GROUP} --group_method $GROUP_METHOD --tensor_parallel_size 1
            cd ..
            mkdir -p ./result_all
            cp -r $exp_dir/result_reflect_sad_${NUM_GROUP}_${exp_tag}_sad/result_all_${exp_tag}/* ./result_all/
            
            python eval_main.py --model $rename_model_name --category normal --language en
            mkdir -p $exp_dir/score_reflect_sad_${NUM_GROUP}_${exp_tag}_$rename_model_name
            cp -r ./score_all/* $exp_dir/score_reflect_sad_${NUM_GROUP}_${exp_tag}_$rename_model_name/
            rm -rf ./score_all
            rm -rf ./result_all
            cp -r ${exp_dir}/data_standard/* ../data_all/
            cd $exp_dir
        done
    done
done