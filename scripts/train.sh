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
: ${2:?"Second argument is the save directory"}
: ${3:?"Third argument is the number of training steps"}

DATA_DIR=$1
SAVE_DIR=$2
TOTAL_UPDATES=$3
LOG_FILE=$SAVE_DIR/training.log

shift 3

mkdir -p $SAVE_DIR


fairseq-train $DATA_DIR \
    --ddp-backend=no_c10d \
    --user-dir $SRC_DIR/morphodropout \
    --save-dir $SAVE_DIR/ \
    --tensorboard-logdir $SAVE_DIR/tb/ \
    --task morpho_translation \
    --arch morpho-transformer \
    --morpho-dropout 0.5 \
    --source-lang src --target-lang tgt \
    --left-pad-source False \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 --min-lr 1e-09 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0\
    --dropout 0.1 \
    --max-tokens 4096 \
    --update-freq 2 --no-progress-bar --log-format simple --log-interval 5 \
    --max-update $TOTAL_UPDATES \
    --save-interval-updates  1000 --keep-interval-updates 20 \
    "$@" \
    2>&1 | tee $LOG_FILE

    #--eval-bleu \
    #--eval-bleu-args '{"beam": 5, "lenpen": 0.6 }' \
    #--eval-bleu-detok moses \
    #--eval-bleu-remove-bpe \
    #--eval-bleu-print-samples \
    #--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
