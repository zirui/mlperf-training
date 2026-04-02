docker run -it --rm \
    --net=host --uts=host \
    --ipc=host --device /dev/dri --device /dev/kfd \
    --security-opt=seccomp=unconfined \
    --volume /data:/data \
    --volume $(pwd):/workspace/code/ \
    --entrypoint /bin/bash \
    --name flux-debug rocm/mlperf:flux1_training_6.0_2026-03-27-20-33-34
