#!/bin/bash

# Check if commit message is provided
if [ $# -eq  ]; then
    echo "Error: Please provide a commit message"
    echo "Usage: $0 \"Your commit message\""
    exit 1
fi

COMMIT_MSG="$1"

# Function to clean up
cleanup() {
    echo "Cleaning up..."
    rm -rf ui model configs components dataset common 2>/dev/null
}

echo "Start copying files..."
cp -r smartcash/model .
cp -r smartcash/configs .
cp -r smartcash/components .
cp -r smartcash/ui .
cp -r smartcash/dataset .
cp -r smartcash/common .

echo "Committing changes..."
if ! git add .; then
    echo "Error: Failed to stage files"
    exit 1
fi

if ! git commit -m "$COMMIT_MSG"; then
    echo "Error: Commit failed"
    exit 1
fi

echo "Pushing to remote..."
if git push origin migration; then
    echo "Push successful"
    cleanup
    exit 0
else
    echo "Error: Push failed. Local changes are kept in the working directory."
    exit 1
fi