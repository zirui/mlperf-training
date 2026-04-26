#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

cd /workspace/code/

python ${SCRIPT_DIR}/download_dataset.py --data_dir /data/gov_report
python ${SCRIPT_DIR}/convert_dataset.py --data_dir /data/gov_report
python ${SCRIPT_DIR}/convert_model.py --output_path=/ckpt/

mv /data/gov_report /data/data
mv /ckpt /data/model/

echo "Data and model preparation completed successfully."
