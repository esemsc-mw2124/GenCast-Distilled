#!/usr/bin/env bash
set -euo pipefail

ZONE="us-south1-a"
TPU_NAME="gencast-mini-vm"   # TPU VM node name
QR_NAME="gencast-mini-vm"    # queued resource name (can differ from TPU_NAME)

# Delete the TPU VM node (this stops charges)
echo "Deleting TPU VM node..."
gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$ZONE" --quiet

# Wait for queued resource to become deletable
echo "Waiting for queued resource to leave ACTIVE state..."
for i in {1..30}; do
    STATE=$(gcloud compute tpus queued-resources describe "$QR_NAME" \
        --zone="$ZONE" --format='value(state)' || true)
    echo "State: $STATE"
    if [[ "$STATE" == "SUSPENDED" || "$STATE" == "FAILED" ]]; then
        break
    fi
    sleep 5
done

# Delete the queued resource
echo "Deleting queued resource..."
gcloud compute tpus queued-resources delete "$QR_NAME" --zone="$ZONE" --quiet
echo "Done."

