CUDA_VISIBLE_DEVICES=0,1 python train_flow.py --name PriOr-RAFT \
                                              --stage City \
                                              --validation City \
                                              --restore_ckpt ./pretrained/raft-things.pth \
                                              --save_path ./checkpoints/City \
                                              --num_steps 60000 \
                                              --batch_size 4 \
                                              --lr 0.0001 \
                                              --wdecay 0.0001 \
                                              --mixed_precision