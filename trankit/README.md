# Training with `trankit`

To run the trankit training, execute the following command:

```bash
    python train.py with \
        train_bio_fpath='./data/train-wo-asia_bibi_.bio' \
        dev_bio_fpath='./data/dev-w-asia_bibi_.bio' \
        -l WARNING \
        -F outputs
```

This will use `sacred` to save the run's configuration into the `outputs`
directory.

If you are trining on multiple GPU devices, the following may be useful:

```bash
    $ export CUDA_VISIBLE_DEVICES=1
```

## Downloading pre-trained `trankit` pipelines

```bash
wget http://nlp.uoregon.edu/download/trankit/bulgarian.zip
unzip bulgarian.zip -d bulgarian

wget http://nlp.uoregon.edu/download/trankit/czech.zip
unzip czech.zip -d czech

wget http://nlp.uoregon.edu/download/trankit/polish.zip
unzip polish.zip -d polish

wget http://nlp.uoregon.edu/download/trankit/russian.zip
unzip russian.zip -d russian

wget http://nlp.uoregon.edu/download/trankit/slovenian.zip
unzip slovenian.zip -d slovenian

wget http://nlp.uoregon.edu/download/trankit/ukrainian.zip
unzip ukrainian.zip -d ukrainian
```

## Generating predictions

```bash
    python predict.py with \
        lang=pl \
        raw_data_dir='../bsnlp2021_train_r1/raw/ryanair/pl/' \
        output_data_dir='./predictions/ryanair/pl/' -F predict_outputs
```

```bash

for lang in bg cs pl ru sl uk;
do
    python predict.py with \
        lang=$lang \
        save_dir='./save_dir_best/' \
        raw_data_dir="../bsnlp2021_train_r1/raw/ryanair/$lang/" \
        output_data_dir="./predictions/ryanair/$lang/" \
        -F predict_outputs;
done
```
