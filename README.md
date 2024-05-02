# ViMedNer

## Setup
```bash
$ python3.7 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Training
```bash
$ python ./run_ner.py \
--data_dir path_to_data \
--model_name_or_path baseline_model \
--labels path_to_label_file \
--output_dir path_to_output_dir \
--max_seq_length 128 \
--num_train_epochs 30 \
--per_device_train_batch_size 16 \
--n_train 1 \
--seed 10 \
--evaluation_strategy "no" \
--logging_strategy "epoch" \
--save_total_limit 2 \
--save_strategy "no" \
--do_train true \
--do_eval true \
--overwrite_output_dir
```

## Evaluation
```bash
$ python ./run_ner.py \
--data_dir path_to_data \
--model_name_or_path path_to_checkpoint \
--labels path_to_label_file \
--output_dir path_to_output_dir \
--max_seq_length 128 \
--num_train_epochs 10 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--seed 11 \
--logging_strategy "epoch" \
--logging_steps 1 \
--load_best_model_at_end True \
--save_total_limit 2 \
--save_strategy "no" \
--do_predict true \
--overwrite_output_dir
```