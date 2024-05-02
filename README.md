# ViMedNer

!python ./train/run_ner.py \
--data_dir path_to_data \
--model_name_or_path baseline_model \
--labels path_to_label_file \
--output_dir . \
--max_seq_length 128 \
--num_train_epochs 30 \
--per_device_train_batch_size 16 \
--n_train 1 \
--seed 10 \
--evaluation_strategy "no" \
--logging_strategy "epoch" \
--save_total_limit 2 \
--save_strategy "no" \
--do_eval true \
--do_predict true \
--overwrite_output_dir
