#!/bin/bash

# === CONFIG ===
PROJECT_ID=gencast-distillation
TPU_NAME=gencast-mini-vm
ZONE=us-south1-a
ACCELERATOR_TYPE=v5litepod-4
RUNTIME_VERSION=v2-tpuv5-litepod

# === CREATE TPU ===
echo "Requesting Spot TPU VM Queued Resource..."
gcloud compute tpus queued-resources create $TPU_NAME \
    --node-id=$TPU_NAME \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --runtime-version=$RUNTIME_VERSION \
    --spot

echo "Waiting for TPU to become ACTIVE..."
# Poll status every 30s until ACTIVE
while true; do
    STATUS=$(gcloud compute tpus queued-resources describe $TPU_NAME --zone=$ZONE --format='value(state)')
    echo "Current status: $STATUS"
    if [ "$STATUS" = "ACTIVE" ]; then
        echo "TPU is ACTIVE! You can now SSH into it."
        break
    elif [ "$STATUS" = "FAILED" ]; then
        echo "TPU provisioning FAILED. Exiting."
        exit 1
    fi
    sleep 30
done

