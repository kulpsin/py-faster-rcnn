#!/bin/bash
# Usage:
# ./experiments/scripts/wider.sh MODE [NET_FINAL] [options args to {train,test}_net.py]
# MODE is either train or test.
#
# Example:
# ./experiments/scripts/wider.sh train

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=0
NET="VGG16"
NET_lc=${NET,,}
DATASET="wider_face"

# Mode either test or train
MODE=$1
NET_FINAL=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

TRAIN_IMDB="wider_face_train"
TEST_IMDB="wider_face_val"
PT_DIR="wider_face"
ITERS=50000

LOG="experiments/logs/wider_faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

case $MODE in
  test)
    time ./tools/test_net.py --gpu ${GPU_ID} \
      --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
      --net ${NET_FINAL} \
      --imdb ${TEST_IMDB} \
      --cfg experiments/cfgs/wider_faster_rcnn_end2end.yml \
      ${EXTRA_ARGS}
    ;;
  train)
    time ./tools/train_net.py --gpu ${GPU_ID} \
      --solver models/${PT_DIR}/${NET}/faster_rcnn_end2end/solver.prototxt \
      --weights data/faster_rcnn_models/${NET}.v2.caffemodel \
      --imdb ${TRAIN_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/wider_faster_rcnn_end2end.yml \
      ${EXTRA_ARGS}
    set +x
    NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
    set -x
    ;;
  *)
    echo "No mode given"
    exit
    ;;
esac
