#!/bin/bash

# Check if commit message is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a commit message"
    echo "Usage: $0 \"Your commit message\""
    exit 1
fi

COMMIT_MSG="$1"
cleanup_needed=0

# Function to clean up
cleanup() {
    if [ $cleanup_needed -eq 1 ]; then
        echo "Cleaning up..."
        rm -rf ui model configs components dataset common 2>/dev/null
    fi
    exit $1
}

# Set up trap to ensure cleanup runs on exit
trap 'cleanup $?' EXIT

echo "Start copying files..."
cp -r smartcash/model .
cp -r smartcash/configs .
cp -r smartcash/components .
cp -r smartcash/ui .
cp -r smartcash/dataset .
cp -r smartcash/common .

# Mark that we need to clean up
cleanup_needed=1

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
    cleanup_needed=0  # Don't clean up if push was successful
    cleanup 0
else
    echo "Error: Push failed. Local changes are still in the working directory."
    exit 1
fi