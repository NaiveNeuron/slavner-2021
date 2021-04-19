#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "We need 3 arguments: Directories for PANX, SLAVNER bio files and OUTPUT."
    echo "Usage: combine_panx_slavner.sh [PANX_DIR] [SLAVNER_DIR] [OUTPUT_DIR]"
    exit 1
fi

panx_dir=$1
slavner_dir=$2
output_dir=$3

for pdir in $(ls -d $panx_dir/*);
do
    lang=$(basename $pdir)
    echo "Running for $lang"
    cat $pdir/train $pdir/dev $pdir/test | sed "s/^$lang:\(.*\)/\1/g" > $pdir/$lang.bio
    cat $slavner_dir/train-wo-ryanair_$lang.bio $pdir/$lang.bio > $output_dir/train-wo-ryanair_$lang.bio
    cp $slavner_dir/dev-w-ryanair_$lang.bio $output_dir
done
