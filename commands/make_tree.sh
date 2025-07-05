#!/bin/bash

# make_tree.sh - Generate a clean tree structure of the smartcash directory
# Usage: ./make_tree.sh [output_file]

OUTPUT_FILE=${1:-"project-structure.txt"}
TARGET_DIR="smartcash"

# Clear output file
echo "Generating directory structure for $TARGET_DIR..." > "$OUTPUT_FILE"

# Function to generate tree
generate_tree() {
    local dir="$1"
    local prefix="$2"
    
    # Get all entries, sorted
    local entries=("$dir"/*)
    local entry_count=${#entries[@]}
    local i=0
    
    for entry in "${entries[@]}"; do
        i=$((i + 1))
        local name=$(basename "$entry")
        
        # Skip hidden files and common directories
        [[ $name == .* ]] && continue
        [[ $name == __pycache__ || $name == .pytest_cache || $name == .git || $name == .idea || $name == .vscode ]] && continue
        
        # Skip common file extensions
        [[ $name == *.pyc || $name == *.pyo || $name == *.pyd || $name == *.so ]] && continue
        [[ $name == *.egg-info || $name == *.pt || $name == *.bin || $name == *.onnx ]] && continue
        
        # Determine if this is the last entry
        local is_last=$((i == entry_count))
        
        # Print current entry
        if [ -z "$prefix" ]; then
            # Root level
            if [ $i -eq 1 ]; then
                echo "$(basename "$TARGET_DIR")" > "$OUTPUT_FILE"
            fi
            echo "├── $name" >> "$OUTPUT_FILE"
        else
            echo "${prefix}├── $name" >> "$OUTPUT_FILE"
        fi
        
        # If directory, recurse
        if [ -d "$entry" ]; then
            local new_prefix="${prefix}│   "
            if [ $is_last -eq 1 ]; then
                new_prefix="${prefix}    "
            fi
            generate_tree "$entry" "$new_prefix"
        fi
    done
}

# Generate the tree
generate_tree "$TARGET_DIR" ""

# Add summary
echo -e "\n# Summary" >> "$OUTPUT_FILE"
echo "Total directories: $(find "$TARGET_DIR" -type d -not -path '*/\.*' -not -path '*__pycache__*' | wc -l | xargs)" >> "$OUTPUT_FILE"
echo "Total files: $(find "$TARGET_DIR" -type f -not -path '*/\.*' -not -path '*__pycache__*' | wc -l | xargs)" >> "$OUTPUT_FILE"

echo "Directory structure generated in $OUTPUT_FILE"