#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "We need 2 arguments: Directories for Slavner input bio files and Stanza NERBASE output directory."
    echo "Usage: combine_panx_slavner.sh [BIO_DIR] [OUTPUT_DIR]"
    exit 1
fi

corpus=$1
output_dir=$2
declare -A code2lang
code2lang=( ["bg"]="Bulgarian" ["ru"]="Russian" ["pl"]="Polish" ["uk"]="Ukrainian" ["sl"]="Slovenian" ["cs"]="Czech" )

for file in $(ls $corpus/train-*);
do
filename=$(basename $file)
dev_file=$(echo $file | sed 's/train-wo-/dev-w-/g')
shorthand=$(echo $filename | grep -oE '(bg|ru|pl|uk|sl|cs)')
lang=${code2lang[$shorthand]}

echo "Writing $lang: $filename | $dev_file"
mkdir -p $output_dir/$lang-Slavner
cp $file $output_dir/$lang-Slavner/train.bio
cp $dev_file $output_dir/$lang-Slavner/dev.bio
cp $dev_file $output_dir/$lang-Slavner/test.bio
done

