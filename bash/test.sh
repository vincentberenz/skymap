#!/bin/bash

# Check if input files exist
for i in {1..5}; do
    if [ ! -f "i$i.tiff" ]; then
        echo "Error: Input file i$i.tiff not found"
        exit 1
    fi
done

# Process each image with solve-field
for i in {1..5}; do
    echo "Processing image $i..."
    solve-field i$i.tiff
    if [ ! -f "i$i.solved" ]; then
        echo "Error: solve-field failed for image $i"
        exit 1
    fi
done

# Create aligned images using proper variable syntax
for i in {1..5}; do
    echo "Aligning image $i..."
    wcs-resample i${i}.new i1.wcs i${i}_aligned.fits
    if [ ! -f "i${i}_aligned.fits" ]; then
        echo "Error: Alignment failed for image $i"
        exit 1
    fi
done

# Extract parameters from first aligned image
echo "Extracting parameters..."
scale=$(wcsinfo -s i1_aligned.fits | awk '{print $1}')
width=$(listhead i1_aligned.fits | grep NAXIS1 | awk '{print $4}')
height=$(listhead i1_aligned.fits | grep NAXIS2 | awk '{print $4}')
center=$(wcsinfo -c i1_aligned.fits)

# Create header template with extracted parameters
echo "Creating header template..."
mHdr -p $scale -h $(echo "scale=$scale; $height/3600" | bc) -w $(echo "scale=$scale; $width/3600" | bc) "$center" template.hdr

# Create image list table
echo "Creating image list table..."
mImgtbl . images.tbl

# Combine images using mAdd
echo "Combining images..."
mAdd images.tbl template.hdr combined.fits

# Verify the final result
if [ -f "combined.fits" ]; then
    echo "Success: combined.fits created"
    wcsinfo combined.fits
else
    echo "Error: Failed to create combined.fits"
    exit 1
fi