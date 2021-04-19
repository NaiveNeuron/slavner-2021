# TraSpaS Stanza


First please follow the installation instruction on (Stanza website)[https://stanfordnlp.github.io/stanza/usage.html].

To train the per language Stanza models you can run following inside `stanza-train/stanza` directory.
```bash
./train_all.sh traspas_submission/
```

To create a training dataset, please use
```bash
copy_slavner_data.sh downloaded_slavner/ stanza-train/data/nerbase
```
