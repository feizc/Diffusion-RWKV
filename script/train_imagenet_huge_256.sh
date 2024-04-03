torchrun --nnodes=1 --nproc_per_node=8 train.py \
--model DRWKV-H/2 \
--dataset-type imagenet \
--data-path /maindata/data/shared/multimodal/public/dataset_img_only/imagenet/data \
--image-size 256 \
--task-type class-cond \
--num-classes 1000 \
--global-batch-size 128 \
--epochs 30 \
--warmup_epochs 0 \
--accum_iter 8 \
--eval_steps 10000000 \
--lr 1e-5 \
--latent_space True \
--global-seed 1314 \
--resume /maindata/data/shared/multimodal/zhengcong.fei/code/diff-rwkv/results/DRWKV-H-2-imagenet-class-cond-256/checkpoints/ckpt5.pth