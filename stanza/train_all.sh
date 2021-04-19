#!/bin/bash


if [ "$#" -ne 1 ]; then
    echo "Usage: train_all.sh [OUTPUT_DIR]"
    exit 1
fi

output_dir=$1; shift

mkdir -p $output_dir
mkdir -p models

# all parameters are set as they were found through grid search
echo "============================== RUNNING Czech =============================="
bash scripts/run_ner.sh Czech-Slavner --max_steps 5000 --num_layers 2 --hidden_dim 256 --char_hidden_dim 256 2>&1 | tee $output_dir/cs.txt

echo "============================== RUNNING Bulgarian =============================="
bash scripts/run_ner.sh Bulgarian-Slavner --max_steps 7000 --num_layers 2 2>&1 | tee $output_dir/bg.txt

echo "============================== RUNNING Polish =============================="
bash scripts/run_ner.sh Polish-Slavner --max_steps 5000 --num_layers 1 --hidden_dim 256 --char_hidden_dim 512 2>&1 | tee $output_dir/pl.txt

echo "============================== RUNNING Russian =============================="
bash scripts/run_ner.sh Russian-Slavner --max_steps 10000 --num_layers 3 2>&1 | tee $output_dir/ru.txt

echo "============================== RUNNING Slovenian =============================="
bash scripts/run_ner.sh Slovenian-Slavner --max_steps 10000 --num_layers 3 2>&1 | tee $output_dir/sl.txt

echo "============================== RUNNING Ukrainian =============================="
bash scripts/run_ner.sh Ukrainian-Slavner --max_steps 5000 --num_layers 3 --hidden_dim 256 --char_hidden_dim 512 2>&1 | tee $output_dir/uk.txt

echo "Storing models"
mkdir -p models/$output_dir
cp saved_models/ner/*.pt models/$output_dir

echo "=========================================================================="
echo "======================= FINAL REPORT BY ITERATION ========================"
echo "+========================================================================="
grep -nH -A 1 'Prec.' $output_dir/*.txt | awk '{print $1, $4, $5, $6}' | sed 's#training_outputs/##g' | sed 's/\.txt[^ ]*//g' | tee models/$output_dir/iterations.txt

echo "=========================================================================="
echo "============================== FINAL REPORT =============================="
echo "=========================================================================="
grep 'Best dev' $output_dir/* | awk '{print $1,"F1="$8}' | sed 's/.txt:[^ ]*//g' | sed "s#$output_dir/##g" | tee models/$output_dir/scores.txt

