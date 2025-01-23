#!/bin/bash

# Check if day number is provided
if [ -z "$1" ]; then
    echo "Usage: ./create-daily-update.sh <day-number> <topic>"
    exit 1
fi

DAY_NUM=$1
TOPIC=$2
FILE_NAME="daily-updates/day-$DAY_NUM-$(echo $TOPIC | tr ' ' '-').md"

# Check if the file already exists
if [ -f "$FILE_NAME" ]; then
    echo "File $FILE_NAME already exists!"
    exit 1
fi

# Create the daily update file from the template
cp daily-updates/template.md "$FILE_NAME"

# Replace placeholders in the template
sed -i "s/Day X:/Day $DAY_NUM:/" "$FILE_NAME"
sed -i "s/\[Topic\]/$TOPIC/" "$FILE_NAME"
sed -i "s/day-X-/day-$DAY_NUM-/" "$FILE_NAME"
sed -i "s/\/src\/day-X-/\/src\/day-$DAY_NUM-/" "$FILE_NAME"

echo "Created $FILE_NAME"
