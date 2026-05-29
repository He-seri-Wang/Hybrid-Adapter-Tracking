# Training BAT
#python tracking/train.py --script bat --config rgbt --save_dir ./output --mode multiple --nproc_per_node 4
#NCCL_P2P_LEVEL=NVL python tracking/train.py --script bat --config rgbt --save_dir ./output --mode multiple --nproc_per_node 3 
#CUDA_VISIBLE_DEVICES=1,2,3 NCCL_P2P_LEVEL=NVL
#CUDA_VISIBLE_DEVICES=1,2,3 NCCL_P2P_LEVEL=NVL \
#python tracking/train.py --script bat --config rgbt --save_dir ./output --mode multiple --nproc_per_node 3 --use_wandb 1
#NCCL_P2P_LEVEL=NVL python tracking/train.py --script bat --config rgbe --save_dir ./output --mode multiple --nproc_per_node 4

#export CUDA_VISIBLE_DEVICES=3
#python tracking/train.py --script bat --config rgbt --save_dir ./output --mode single

#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=0   
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#torchrun --nproc_per_node=4 \
#tracking/train.py \
#  --script bat \
#  --config rgbt \
#  --save_dir ./output \
#  --mode multiple

CUDA_VISIBLE_DEVICES=1,3 NCCL_P2P_LEVEL=NVL \
python tracking/train.py --script bat --config rgbt --save_dir ./output --mode multiple --nproc_per_node 2 --use_wandb 1