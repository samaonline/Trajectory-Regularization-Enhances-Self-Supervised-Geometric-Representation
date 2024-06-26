python3 main_linear.py \
    --dataset $1 \
    --backbone resnet18 \
    --train_data_path ./datasets \
    --val_data_path ./datasets \
    --max_epochs 400 \
    --devices 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer lars \
    --exclude_bias_n_norm_lars \
    --scheduler warmup_cosine \
    --lr 1.0 \
    --weight_decay 1e-4 \
    --batch_size 256 \
    --num_workers 4 \
    --crop_size 32 \
    --name vicreg-$1 \
    --pretrained_feature_extractor $2 \
    --save_checkpoint \
    --checkpoint_dir traj/ood_linear_conv4_1 \
    --auto_resume \
    --num_classes 50 \
    --wandb \
    --entity 'YOUR_WANDB' \
    --checkpoint_frequency 20