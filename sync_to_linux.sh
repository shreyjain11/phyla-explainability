#!/bin/bash
# Sync Phyla_MambaLRP_Integration to Linux GPU server
# Usage: ./sync_to_linux.sh

set -e

MACOS_DIR="/Users/shreyjain/Downloads/Phyla_MambaLRP_Integration"
LINUX_SERVER="shrey@34.172.92.250"
LINUX_DIR="~/work"

echo "=================================="
echo "Syncing to Linux GPU Server"
echo "=================================="
echo "Source: $MACOS_DIR"
echo "Target: $LINUX_SERVER:$LINUX_DIR"
echo ""

# Show what will be synced
echo "Files to sync:"
echo "  - Phyla/phyla/model/model.py (UPDATED - model loading fixes)"
echo "  - scripts/comprehensive_validation.py (NEW - validation script)"
echo "  - integrations/* (existing adapters)"
echo ""

# Perform the sync
echo "Starting rsync..."
rsync -avz --progress \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='*.egg-info' \
  --exclude='outputs/' \
  "$MACOS_DIR/" \
  "$LINUX_SERVER:$LINUX_DIR/"

echo ""
echo "✅ Sync complete!"
echo ""
echo "Next steps on Linux:"
echo "  1. cd ~/work"
echo "  2. python scripts/comprehensive_validation.py"
echo ""

