#!/bin/bash

# python eval_real.py --models mcd_unet --aggregation kalman --mc-samples 50 \
#   --checkpoint-template checkpoint_{model}.pth --benchmark \

python eval_real.py --models mcd_unet --aggregation maj_vote --mc-samples 50 \
  --checkpoint-template checkpoint_{model}.pth --benchmark \

# python eval_real.py --models mcd_unet --aggregation post_avg --mc-samples 50 \
#   --checkpoint-template checkpoint_{model}.pth --benchmark \

# python eval_real.py --models resnet,sliding_max \
#     --checkpoint-template checkpoint_{model}.pth --benchmark \

# python eval_real.py --models unet,mcd_unet,mobilenet,performer,transformer,longformer \
#     --checkpoint-template checkpoint_{model}.pth --benchmark \

python eval_real.py --models sliding_max \
    --checkpoint-template checkpoint_{model}.pth --benchmark \
