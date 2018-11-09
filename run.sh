#!/bin/sh

export PYTHONPATH="$PYTHONPATH:/home/yincui/tpu/models"

STORAGE_BUCKET=gs://tpu_training
DATASET=ILSVRC2012
DATA_DIR=${STORAGE_BUCKET}/data/${DATASET}
RESNET_DEPTH=50
TRAIN_STEPS=112603
TRAIN_BATCH_SIZE=1024
EVAL_BATCH_SIZE=1000
NUM_TRAIN_IMAGES=1281167
NUM_EVAL_IMAGES=50000
NUM_CLASSES=1000
STEPS_PER_EVAL=2502
ITER_PER_LOOP=1251
LR=0.1
LOG_STEPS=100
BETA=0.999
GAMMA=1.0

MODEL_DIR=${STORAGE_BUCKET}/${DATASET}_resnet${RESNET_DEPTH}_${BETA}_${GAMMA}

python /usr/share/tpu/models/official/resnet/resnet_main.py \
  --tpu=${TPU_NAME} \
  --data_dir=${DATA_DIR} \
  --model_dir=${MODEL_DIR} \
  --resnet_depth=${RESNET_DEPTH} \
  --train_steps=${TRAIN_STEPS} \
  --train_batch_size=${TRAIN_BATCH_SIZE} \
  --eval_batch_size=${EVAL_BATCH_SIZE} \
  --num_train_images=${NUM_TRAIN_IMAGES} \
  --num_eval_images=${NUM_EVAL_IMAGES} \
  --num_label_classes=${NUM_CLASSES} \
  --steps_per_eval=${STEPS_PER_EVAL} \
  --iterations_per_loop=${ITER_PER_LOOP} \
  --base_learning_rate=${LR} \
  --log_step_count_steps=${LOG_STEPS} \
  --beta=${BETA} \
  --gamma=${GAMMA}
