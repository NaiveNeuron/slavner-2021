# slavner-2021

A set of tools for reproduction of the results of TraSpaS at the BSNLP2021 Shared Task.

## Data

To download the data, run the following

    $ wget http://bsnlp.cs.helsinki.fi/data/bsnlp2021_train_r1.zip
    $ unzip bsnlp2021_train_r1.zip

## Generate `.csv` dataset

Run following in order to generate the `.csv` dataset called `slavner-2019-preprocessed.csv`
from dataset `./bsnlp2021_train_r1` using `nltk` tokenizer:
```bash
    python3 preprocess.py slavner-2019-preprocessed.csv \
        --input-path ./bsnlp2021_train_r1 \
        --tokenizer nltk
```

## Generate `.bio` files

In order to generate the train/valid `.bio` files from the preprocessed
`slavner-2019-preprocessed.csv` file, using `ryanair` as the validation
"topic" and output the `tags` column:

```bash
    python3 generate-bio.py \
        slavner-2019-preprocessed.csv \
        train-wo-ryanair.bio \
        dev-w-ryanair.bio \
        --validation-topic ryanair \
        --output-column tags
```

You can also generate the BIO for unique values in specific column. For example,
using the above example but split by language:
```bash
    python3 generate-bio.py \
        slavner-2019-preprocessed.csv \
        train-wo-ryanair.bio \
        dev-w-ryanair.bio \
        --validation-topic ryanair \
        --output-column tags
        --split-by lang
```
This will generate file in format `train-wo-ryanair_{lang}.bio` (e.g., `train-wo-ryanair_cs.bio`).

## Reproducing resuts of respective models

More information on the reproduction of the considered models can be found in
the following subfolders:

- [Trankit](trankit/)
- [Stanza](stanza/)

## Paper

More information can be found in the paper called [Benchmarking Pre-trained Language Models for Multilingual NER: TraSpaS at the BSNLP2021 Shared Task](https://www.aclweb.org/anthology/2021.bsnlp-1.13/)

Should any part of this repository be useful, please cite:

```
@inproceedings{suppa-jariabka-2021-benchmarking,
    title = "Benchmarking Pre-trained Language Models for Multilingual {NER}: {T}ra{S}pa{S} at the {BSNLP}2021 Shared Task",
    author = "Suppa, Marek  and
      Jariabka, Ondrej",
    booktitle = "Proceedings of the 8th Workshop on Balto-Slavic Natural Language Processing",
    month = apr,
    year = "2021",
    address = "Kiyv, Ukraine",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.bsnlp-1.13",
    pages = "105--114"
}
```
