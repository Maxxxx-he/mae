TIME=$(date +%s%3N)
OUTPUT_DIR='/home/h_haoy/Documents/GitHub/mae'
python -m torch.distributed.launch --nproc_per_node=1 main_pretrain.py \
        --model mae_vit_base_patch16 \
        --input_size 224 \
        --batch_size 64 \
        --mask_ratio 0.75 \
        --warmup_epochs 40 \
        --epochs 200 \
        --blr 1e-3 \
        --output_dir ${OUTPUT_DIR}  \
        --dist_url "file://$OUTPUT_DIR/$TIME"
