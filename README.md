# GPTorch
Implementation of GPT from Scratch.
install datasets,torchinfo,wandb,tiktoken

bash torchrun --standalone \ --nproc_per_node=4 \ train.py --config config.yaml
torchrun --standalone --nproc_per_node=2 blog/test_train.py 


CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=2 blog/debug_ddp.py 