PEAK_LR=0.0006          # Peak learning rate, adjust as needed
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
DATA_DIR=./data-bin
#USER_DIR=$1
ARCH=$1
SAVE_DIR=log/${ARCH}_v$2
mkdir -p $SAVE_DIR
TENSORBOARD_LOGDIR=$SAVE_DIR/tensorboard_logs
mkdir -p $TENSORBOARD_LOGDIR

ENCODER_LAYERS=12
ENCODER_MIX_LAYERS=12

#TOTAL_UPDATES=28569
#WARMUP_UPDATES=2856
#TOKENS_PER_SAMPLE=128
#MAX_SENTENCES=64
#UPDATE_FREQ=32
#
#
#python train.py $DATA_DIR \
#    --task masked_lm --criterion masked_lm \
#    --arch ${ARCH} --sample-break-mode complete \
#    --tokens-per-sample $TOKENS_PER_SAMPLE \
#    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 5.0 \
#    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
#    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
#    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
#    --skip-invalid-size-inputs-valid-test \
#    --tensorboard-logdir $TENSORBOARD_LOGDIR \
#    --num-workers 20 --ddp-backend=no_c10d \
#    --save-dir $SAVE_DIR --keep-interval-updates 4 \
#    --max-update $TOTAL_UPDATES --log-format simple --log-interval 10 --no-epoch-checkpoints\
#    --encoder-layers $ENCODER_LAYERS --encoder-mix-layers $ENCODER_MIX_LAYERS \
#    #--user-dir $USER_DIR \


TOTAL_UPDATES=500000
WARMUP_UPDATES=24000
TOKENS_PER_SAMPLE=512
MAX_SENTENCES=16
UPDATE_FREQ=128


python train.py $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --arch ${ARCH} --sample-break-mode complete \
    --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --tensorboard-logdir $TENSORBOARD_LOGDIR \
    --num-workers 20 --ddp-backend=no_c10d \
    --save-dir $SAVE_DIR --keep-interval-updates 4 \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 5 --no-epoch-checkpoints \
    --encoder-layers $ENCODER_LAYERS --encoder-mix-layers $ENCODER_MIX_LAYERS \
    --reset-lr-scheduler --reset-optimizer \
    # --user-dir $USER_DIR \
