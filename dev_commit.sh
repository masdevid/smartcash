#!/bin/bash

# Check if commit message is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a commit message"
    echo "Usage: $0 \"Your commit message\""
    exit 1
fi

COMMIT_MSG="$1"

echo "Start copying files..."
cp -r smartcash/model .
cp -r smartcash/configs .
cp -r smartcash/components .
cp -r smartcash/ui .
cp -r smartcash/dataset .
cp -r smartcash/common .

sleep 1

echo "Committing changes..."
git add .
git commit -m "$COMMIT_MSG"

echo "Pushing to remote..."
git push origin migration

sleep 3

echo "Cleaning up..."
rm -rf ui model configs components dataset common 2>/dev/null

echo "Done!"