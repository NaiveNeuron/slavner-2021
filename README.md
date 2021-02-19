# slavner-2021
A set of tools for SlavNER 2021


## Data

To download the data, run the following

    $ wget http://bsnlp.cs.helsinki.fi/data/bsnlp2021_train_r1.zip
    $ unzip bsnlp2021_train_r1.zip

## Generate `.csv` dataset

Run following in order to generate the `.csv` dataset called `slavner-2019-preprocessed.csv`
using nltk tokenizer:
```bash
    python3 preprocess.py slavner-2019-preprocessed.csv --tokenizer nltk
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
