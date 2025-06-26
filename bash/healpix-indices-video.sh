#!/bin/bash

# Set input/output paths
INPUT_DIR="/home/vberenz/Workspaces/skymap-data/june_19th_2025/ns3/healpix"
OUTPUT_FILE="/home/vberenz/Workspaces/skymap-data/june_19th_2025/ns3/video/healpix-indices.mp4"

# Debug: Show directory contents
echo "Directory contents:"
ls -l "$INPUT_DIR"

# Debug: Show exact pattern matching
echo "Files matching pattern:"
ls -l "$INPUT_DIR/nightskycam3_*.pkl.overlayed.tiff"

# Verify input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Get sorted list of files
files=($(find "$INPUT_DIR" -name 'nightskycam3_*.overlayed.tiff' -printf '%p\n' | sort -t '_' -k 2))

# Debug: Show sorted files
echo "Sorted files:"
echo "$files"

# Verify files exist
if [ -z "$files" ]; then
    echo "Error: No TIFF files found in $INPUT_DIR"
    exit 1
fi

# Create video using FFmpeg
ffmpeg -framerate 8 \
       -pattern_type glob -i "$INPUT_DIR/nightskycam3_*.pkl.overlayed.tiff" \
       -vf "scale=iw/4:-1" \
       -c:v libx264 -crf 18 \
       "$OUTPUT_FILE"
