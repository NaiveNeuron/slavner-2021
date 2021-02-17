# slavner-2021
A set of tools for SlavNER 2021


## Generate `.bio` files

In order to generate the train/valid `.bio` files from the preprocessed
`slavner-2019-preprocessed.csv` file, using `ryanair` as the validation
"topic" and output the `tags` column:

    python3 generate-bio.py slavner-2019-preprocessed.csv train-wo-ryanair.bio dev-w-ryanair.bio --validation-topic ryanair --output-column tags
