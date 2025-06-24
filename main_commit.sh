#!/bin/bash

# Get current datetime in a readable format
CURRENT_DATETIME=$(date +"%Y-%m-%d %H:%M:%S")

# Commit message template
COMMIT_MSG="Published at ${CURRENT_DATETIME}"

# Clean up smartcash directory if it exists
if [ -d "smartcash" ]; then
    rm -rf smartcash
fi

# Add all changes
git add .

# Commit with the formatted message
git commit -am "${COMMIT_MSG}"

# Force push to main branch
git push origin main --force

# Switch back to migration branch
git checkout migration

echo "Production deployment completed at ${CURRENT_DATETIME}"