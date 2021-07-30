#!/usr/bin/env bash
set -euo pipefail

echo "Tactile data is being downloaded ..."
./gdrivedl.py https://drive.google.com/drive/folders/1oDegTXkPEdRmIgA-b69fs60qJlSlxjGG?usp=sharing ../data_temp
