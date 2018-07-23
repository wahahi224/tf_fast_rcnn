#!/bin/bash

GPU_ID=$1 

NET="vgg16"

TRAIN_IMDB="voc_2007_trainval"
TEST_IMDB="voc_2007_test"
ITERS=70000



FILE_TAG="test_fast_rcnn_0422"
NET_FINAL="output/default/voc_2007_trainval/vgg16_fast_rcnn_0422/res101_faster_rcnn_iter_110000.ckpt"



LOG="experiments/logs/${FILE_TAG}_vgg16.log.`date +'%Y-%m-%d_%H-%M-%S'`"
CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net_tf.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --tag ${FILE_TAG} \
    --net ${NET} \
    --set \
    2>&1 | tee "$LOG"


