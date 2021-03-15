# Run allennlp training locally

#
# edit these variables before running script
DATASET='ebmnlp'
TASK='pico'
ELMO='elmo_htlm_reparam_hypeL1_0.5'
with_finetuning='' #'_finetune'  # or '' for not fine tuning
dataset_size=38124
export EXPERIMENT_DIR='/workspace/umls/umls_explorer/experiments/test/'"$ELMO"
OUTPUT_DIR=$EXPERIMENT_DIR/$TASK/$DATASET

export ELMO_WEIGHTS=$EXPERIMENT_DIR/model.tar.gz

export DATASET_SIZE=$dataset_size

CONFIG_FILE=allennlp_config/"$TASK""$with_finetuning".json

SEED=13270
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export IS_LOWERCASE=false
export TRAIN_PATH=data/tasks/$TASK/$DATASET/train.txt
export DEV_PATH=data/tasks/$TASK/$DATASET/dev.txt
export TEST_PATH=data/tasks/$TASK/$DATASET/test.txt

export CUDA_DEVICE=3

export GRAD_ACCUM_BATCH_SIZE=32
export NUM_EPOCHS=5
export LEARNING_RATE=0.001

allennlp train $CONFIG_FILE  --include-package elmo_hype -s $OUTPUT_DIR