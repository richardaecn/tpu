#!/bin/sh

export PYTHONPATH="$PYTHONPATH:/home/yincui/tpu/models"

STORAGE_BUCKET=gs://tpu_training
DATA_DIR=${STORAGE_BUCKET}/data/ILSVRC2012
MODEL_DIR=${STORAGE_BUCKET}/resnet50_softmax
ITER_PER_LOOP=200

python /usr/share/tpu/models/official/resnet/resnet_main.py \
  --tpu=$TPU_NAME \
  --data_dir=${DATA_DIR} \
  --model_dir=${MODEL_DIR} \
  --iterations_per_loop=${ITER_PER_LOOP}
