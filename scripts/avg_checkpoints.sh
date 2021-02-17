#!/bin/bash


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath $SCRIPT_DIR/..)
THIRDP_DIR=$(realpath $ROOT_DIR/3party)
SRC_DIR=$(realpath $ROOT_DIR/src)

. ~/miniconda3/etc/profile.d/conda.sh
conda activate nlp_pytorch

: ${1:?"First argument is the checkpoint directory"}

CHECKPOINT_DIR=$1

shift

python $THIRDP_DIR/fairseq/scripts/average_checkpoints.py \
   --input $CHECKPOINT_DIR \
   --output $CHECKPOINT_DIR/avg.pt  \
   --num-update-checkpoints 5

