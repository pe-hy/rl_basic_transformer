source activate rl
CUDA_VISIBLE_DEVICES=1 parallel python train.py --batch_size 64 --vocab_size 106 --block_size 20 --n_embd 96 --n_hidden 16 --datapath /mnt/raid/data/Hyner_Petr/rl/rl_basic_transformer/Data/data_cube.pkl --n_iters {1} --init_bottleneck_by_last {2} --divider {3} ::: 3 4 ::: False ::: 2 4 5 10