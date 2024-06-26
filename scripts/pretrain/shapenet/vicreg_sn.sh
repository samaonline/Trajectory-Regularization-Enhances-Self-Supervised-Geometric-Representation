python3 main_pretrain.py \
    --dataset $1 \
    --backbone resnet18 \
    --train_data_path ./datasets \
    --val_data_path ./datasets \
    --max_epochs 180 \
    --devices 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm_lars \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 300 \
    --num_workers 4 \
    --crop_size 32 \
    --min_scale 0.2 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --solarization_prob 0.1 \
    --gaussian_prob 0.0 0.0 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name vicreg-$1 \
    --project geo_pretrain \
    --entity unitn-mhug \
    --save_checkpoint \
    --checkpoint_dir traj/has_0.3_a0.01 \
    --auto_resume \
    --method vicreg \
    --proj_hidden_dim 2048 \
    --proj_output_dim 2048 \
    --sim_loss_weight 25 \
    --var_loss_weight 25.0 \
    --cov_loss_weight 1.0 \
    --checkpoint_frequency 1 \
    --wandb \
    --entity 'YOUR_WANDB' \
    --alpha 0.01
