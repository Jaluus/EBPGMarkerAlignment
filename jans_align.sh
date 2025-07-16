#! /bin/bash
# Jans Align Script

SCRIPT_HOME="/home/pg/users/bsuo/jan/"

source $SCRIPT_HOME/markers/bin/activate
python3 $SCRIPT_HOME/EBGPMarkerAlignment/alignment.py \
    --step-size-x 64 \
    --step-size-y 64 \
    --resolution-x 512 \
    --resolution-y 512 \
    --sample-average-exponent 2 \
    --frame-average-exponent 2 \
    --iterations 5
    