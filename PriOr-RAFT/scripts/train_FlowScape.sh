CUDA_VISIBLE_DEVICES=0,1,2  python train_flow.py --name PriOr-RAFT \
                                                 --stage FlowScape \
                                                 --validation FlowScape \
                                                 --restore_ckpt ./pretrained/raft-things.pth \
                                                 --save_path ./checkpoints/FlowScape \
                                                 --num_steps 100000 \
                                                 --batch_size 6 \
                                                 --lr 0.0001 \
                                                 --wdecay 0.0001 \
                                                 --mixed_precision