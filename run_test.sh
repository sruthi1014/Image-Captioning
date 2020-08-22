#!/bin/sh
#############################################################################
# TODO: Initialize anything you need for the forward pass
#############################################################################
python -u test.py \
    --model test \
    --epochs 1 \
    --weight-decay 0.0 \
    --momentum 0.0 \
    --batch-size 32 \
    --optimizer adam \
    --hidden-dim 512 \
    --embed-dim 256 \
    --checkpoint iter3_2 \
    --lr 0.0001 | tee test.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
