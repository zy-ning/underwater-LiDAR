#!/bin/bash

# rename "mm" to "cm" in all names of files under data/ recursively

find data/ -type f -name "*mm*" | while read file; do
    newfile=$(echo "$file" | sed 's/mm/cm/g')
    mv "$file" "$newfile"
done

