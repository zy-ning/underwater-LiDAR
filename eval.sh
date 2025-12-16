#!/bin/bash

python eval_real.py --models mcd_unet --aggregation kalman --mc-samples 50 \
  --checkpoint-template checkpoint_{model}.pth --log-dir data/20251213/

# python eval_real.py --models resnet --aggregation maj_vote \
#     --log-dir data/20251213/

