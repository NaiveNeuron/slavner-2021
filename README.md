# slavner-2021
A set of tools for SlavNER 2021


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
