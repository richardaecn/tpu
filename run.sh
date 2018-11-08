#!/bin/sh

STORAGE_BUCKET=gs://tpu_training
DATA_DIR=${STORAGE_BUCKET}/data/ILSVRC2012/

python3 ./tpu/models/official/resnet/resnet_main.py \
  --tpu=${TPU_NAME} \
  --data_dir=${DATA_DIR} \
  --model_dir=${STORAGE_BUCKET}/resnet50_softmax
