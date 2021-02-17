#!/bin/bash


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath $SCRIPT_DIR/..)
THIRDP_DIR=$(realpath $ROOT_DIR/3party)
SRC_DIR=$(realpath $ROOT_DIR/src)

. ~/miniconda3/etc/profile.d/conda.sh
conda activate nlp_pytorch

export PYTHONPATH=$THIRDP_DIR/subword-nmt:$THIRDP_DIR/seqp:$SRC_DIR
export HDF5_USE_FILE_LOCKING=FALSE

: ${1:?"First argument is the data directory"}
: ${2:?"Second argument is the checkpoint"}

DATA_DIR=$1
CHECKPOINT=$2

shift 2

# Using --morpho-dropout 0.0 is not needed, as it is enforced in
# the code for validation and test, but we use it here anyway

fairseq-generate $DATA_DIR \
    --user-dir $SRC_DIR/morphodropout \
    --task morpho_translation \
    --path $CHECKPOINT \
    --morpho-dropout 0.0 \
    --source-lang src --target-lang tgt \
    --left-pad-source False \
    --gen-subset test --beam 5 --lenpen 1.2 --remove-bpe \
    "$@"

