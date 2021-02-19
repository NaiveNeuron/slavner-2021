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
