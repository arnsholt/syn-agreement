#!/bin/bash

for f in */*.tree; do
    echo "$(basename $f)" | sed 's/-.\+//'
done | sort | uniq -c
