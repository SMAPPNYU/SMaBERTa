export TRAIN_FILE=./data/lm_train
export TEST_FILE=./data/lm_test

python3 lm_finetuning.py \
    --output_dir=output \
    --model_type=roberta-base \
    --model_name_or_path=roberta-base \
    --line_by_line \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --mlm \
    --eval_data_file=$TEST_FILE

