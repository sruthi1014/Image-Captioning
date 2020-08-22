#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u train.py \
    --model Pretrained \
    --epochs 15\
    --weight-decay 0.0 \
    --momentum 0.99 \
    --batch-size 32 \
    --optimizer adam \
    --hidden-dim 512 \
    --embed-dim 256 \
    --lr 0.005| tee test32_0.005.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################