python run.py \
    --output_dir="checkpoint" \
    --text_model_name_or_path="klue/roberta-large" \
    --vision_model_name_or_path="openai/clip-vit-base-patch32" \
    --tokenizer_name="klue/roberta-large" \
    --train_file="../data/coco/train_annotations.json" \
    --validation_file="../data/coco/valid_annotations.json" \
    --do_train --do_eval \
    --num_train_epochs="40" --max_seq_length 96 \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --preprocessing_num_workers 32

    ## --run_from_checkpoint="pre_trained"
