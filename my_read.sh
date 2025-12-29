# TRAINING COMMAND FOR GTX 1650 (READ DATASET)
# - train-bs 2 (Low VRAM)
# - workers 0 (Windows Fix)
# - val-bs 4 (Safety)

python train.py --exp-name read \
--max-lr 1e-3 \
--train-bs 2 \
--val-bs 4 \
--num-workers 0 \
--weight-decay 0.5 \
--mask-ratio 0.4 \
--attn-mask-ratio 0.1 \
--max-span-length 8 \
--img-size 512 64 \
--proj 8 \
--dila-ero-max-kernel 2 \
--dila-ero-iter 1 \
--proba 0.5 \
--alpha 1 \
--total-iter 100000 \
READ