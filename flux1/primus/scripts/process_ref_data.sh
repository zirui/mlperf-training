#!/bin/bash

: "${IMAGE:?IMAGE not set}"
: "${DATADIR:=/data/mlperf_flux1/}"
: "${CONT_NAME:=mlperf_flux1}"

docker run --rm --init --detach \
    --net=host --uts=host \
    --ipc=host --device /dev/dri --device /dev/kfd \
    --security-opt=seccomp=unconfined \
    --volume "${DATADIR}":/dataset/ \
    --volume $(pwd):/workspace/code/ \
    --name "${CONT_NAME}" "${IMAGE}" sleep infinity

docker exec "${CONT_NAME}" mkdir /dataset/energon
docker exec "${CONT_NAME}" python /workspace/code/scripts/to_webdataset.py --input_path /dataset/cc12m_preprocessed --output_path /dataset/energon/train --num_workers 8
docker exec "${CONT_NAME}" python /workspace/code/scripts/to_webdataset.py --input_path /dataset/coco_preprocessed --output_path /dataset/energon/val --num_workers 8
docker exec "${CONT_NAME}" python /workspace/code/scripts/energon_prepare.py /dataset/energon --num-workers 8 --template-dir /workspace/code/scripts/dataset_template/
docker exec "${CONT_NAME}" cp -r /dataset/empty_encodings /dataset/energon/empty_encodings

docker container rm -f "${CONT_NAME}"
