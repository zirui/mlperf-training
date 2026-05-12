## Running AMD text-to-image FLUX PyTorch MLPerf Benchmark

This file contains the instructions for running the AMD text-to-image FLUX PyTorch MLPerf Benchmark on AMD hardware.

## 1. Hardware Requirements

- At least 6TB disk space is required.
- GPUs are not required for dataset preparation.

## 2. Software Requirements

- [Docker](https://www.docker.com/)
- Slurm for multinode training

## 3. Set up

### 3.1 Build the container

Build the image and tag it for your docker registry:

```bash
docker build -t <docker-registry>/amd-mlperf:flux1_training_6.0 .
```

Make sure the image is accessible on every node of the intended run environment (especially for multi-node runs).

### 3.2 Download dataset and preprocess

The dataset download/preprocessing scripts are included in the container. To invoke them, you need either a docker or slurm/enroot environment. Start the container, replacing `</path/to/dataset>` with the existing path to where you want to save the dataset and the model weights/tokenizer:

```bash
docker run -it --rm --network=host --ipc=host --volume </path/to/dataset>:/dataset rocm/amd-mlperf:flux1_training_6.0

# now you should be inside the container in the /workspace/flux directory
pip install datasets # install hf_datasets

# download dataset and empty encodings
cd /dataset
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri

# convert to webdataset format
mkdir energon
python /workspace/code/scripts/to_webdataset.py --input_path /dataset/cc12m_preprocessed --output_path /dataset/energon/train --num_workers 8
python /workspace/code/scripts/to_webdataset.py --input_path /dataset/coco_preprocessed --output_path /dataset/energon/val --num_workers 8

# prepare energon metadata
cd energon
energon prepare --split-parts 'train:train/.*' --split-parts 'val:val/.*' ./
# Select y for duplicate keys
# Select y for creating interactively
# Select class 11

# copy over empty_encodings
cp -r ../empty_encodings .

# (Optional) to reclaim space delete /dataset/cc12m_preprocessed and /dataset/coco_preprocessed
```

After, the data structure should look like:
```
/dataset
├── energon
│   ├── train
│   ├── val
│   └── empty_encodings
```

Exit the container.

## 4. Launch training (single node)

### Set Environment

Set the directories for the data and results. Make sure `$LOGDIR` exists and is writable by the container UID (`mkdir -p "$LOGDIR" && sudo chmod 777 "$LOGDIR"`). In this example, we use `/data/mlperf_flux1/results` as the results directory, so please make sure to create this directory.

```bash
export DATADIR=/data/mlperf_flux1/data
export LOGDIR=/data/mlperf_flux1/results
export CONT=<docker-registry>/amd-mlperf:flux1_training_6.0
```

### Set Configuration

Source the appropriate config for your GPU and precision:

```bash
source config_MI355X_01x08x64.sh
```

### Launch a single training run
If you want to perform a single run, use:
```bash
sudo chmod -R 777 .
export NEXP=1
bash run_with_docker.sh
```
## 5. Launch training (multi-node via slurm)

Use the job.sh in the mlperf-training/flux1/nemo folder to launch the training run (update job.sh per your cluster/run configuration).
```
sbatch job.sh
```

Note: For MI355X multi-node runs, we are using the AINIC bundle version 1.117.5-a-77.

## 6. Evaluation

### Quality metric
Validation loss averaged over 8 equidistant time steps [0, 7/8], as described in [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2403.03206).
The validation dataset is prepared in advance so that each sample is associated with a timestep.
This is an integer from 0 to 7 inclusive, and thus should be divided by `8.0` to obtain the timestep.

The algorithm is as follows:

```pseudocode
ALGORITHM: Validation Loss Computation

INPUT:
  - validation_samples: set of validation data samples

INITIALIZE:
  - sum[8]: array of zeros for accumulating losses
  - count[8]: array of zeros for counting samples per timestep

FOR each sample, timestep in validation_samples:
    loss = forward_pass(sample, timestep=t/8)
    sum[t] += loss
    count[t] += 1

mean_per_timestep = sum / count
validation_loss = mean(mean_per_timestep)

RETURN validation_loss
```

As we ensure that the validation set has an equal number of samples per timestep, 
a simple average of all loss values is equivalent to the above.

### Quality target
0.586
### Evaluation frequency
Every 262,144 training samples.
### Evaluation thoroughness
29,696 samples
