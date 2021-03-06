# Training with `spacy`

```bash
python -m spacy init config base_config.cfg --lang pl --pipeline ner --optimize accuracy --gpu -F
python -m spacy init fill-config base_config.cfg config.cfg
cat train-wo-ryanair.bio | sed ':a;N;$!ba;s/\n\n/$$$/g' | tr '\n' ' ' | tr '\t' '|' |  sed 's/\$\$\$/\n/g' > spacified.train-wo-ryanair.bio
cat dev-w-ryanair.bio | sed ':a;N;$!ba;s/\n\n/$$$/g' | tr '\n' ' ' | tr '\t' '|' |  sed 's/\$\$\$/\n/g' > spacified.dev-w-ryanair.bio
```

## Generating predictions

```bash
    python predict.py with \
        lang=pl \
        model_path=pl/spacy_outputs/model-best \
        raw_data_dir='../bsnlp2021_train_r1/raw/ryanair/pl/' \
        output_data_dir='./predictions-spacy/ryanair/pl/' -F predict_spacy_outputs
```

