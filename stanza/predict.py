import pandas as pd
import numpy as np
from pathlib import Path
from dataset import SlavNERDataset
import stanza
from tqdm import tqdm
import re
import os
import typer


eregex = re.compile(r'E-(.*)')
sregex = re.compile(r'S-(.*)')


def bioes2bio(token):
    return sregex.sub(r'B-\1', eregex.sub(r'I-\1', token))


def get_token_info(predictions):
    tokens, words, lemmas = [], [], []
    for sent in predictions.sentences:
        tokens.append([bioes2bio(token.ner) for token in sent.tokens])
        words.append([token.text for token in sent.tokens])
        lemmas.append([token.lemma for token in sent.words])
    return tokens, words, lemmas


def get_final_tag(tags):
    # If we matched a token with both ORG and PER, return PER
    # http://bsnlp.cs.helsinki.fi/System_response_guidelines-1.2.pdf
    # (page 3)
    if 'ORG' in tags and 'PER' in tags:
        return 'PER'

    # If we matched a token with both ORG and PRO, return PRO
    # http://bsnlp.cs.helsinki.fi/System_response_guidelines-1.2.pdf
    # (page 3)
    elif 'ORG' in tags and 'PRO' in tags:
        return 'ORG'
    return tags[0].split('-')[1]


def bio2slavner(row):
    output = []
    i = 0
    row += [('END', 'END', 'B-END')]
    while i < len(row):
        word, lemma, tag = row[i]
        shift = 1
        if tag.startswith('B-'):
            for shift, item in enumerate(row[i+1:], 1):
                if not item[2].startswith('I-'):
                    break
            word =  ' '.join([x[0] for x in row[i:i+shift]])
            lemma = ' '.join([x[1] if x[1] is not None else x[0] for x in row[i:i+shift]])
            tags = [x[2] for x in row[i:i+shift]]
            i += shift
        else:
            i += 1
        output_tag = get_final_tag(tags)
        output.append((word, lemma, output_tag))
    return output[:-1]


def write_predictions(df, lang, topic, file_id, output_path):
    just_id = file_id.split('-')[-1]
    filename = output_path / f'{topic}_{lang}.txt_file_{just_id}.out'
    with open(filename, 'w') as f:
        f.write(f'{file_id}\n')
        for row in df.itertuples():
            line = [items for items in zip(row.words, row.lemmas, row.preds) \
                    if items[2] != 'O']
            if not line:
                continue
            line = bio2slavner(line)
            for word, lemma, tag in line:
                f.write(f'{word}\t{lemma}\t{tag}\tORG-RAND\n')


def main(model_path: Path = typer.Argument(
        ...,
        exists=False,
        dir_okay=True,
        file_okay=False,
        readable=True
    ),
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=True,
        file_okay=False,
        readable=True
    ),
    output_path : Path = typer.Option(
        "./predictions",
        dir_okay=True,
        file_okay=False,
        readable=True
    )
):
    dataset = SlavNERDataset(input_path, None, None)
    os.makedirs(output_path, exist_ok=True)

    for topic_dir in input_path.glob('*'):
        topic = topic_dir.name
        for lang_dir in topic_dir.glob('*'):
            lang = lang_dir.name
            mpath = f'{model_path}/{lang}_slavner_nertagger.pt'
            nlp = stanza.Pipeline(lang=lang, processors='tokenize,ner,lemma',
                                  ner_model_path=str(mpath))
    
            output_dir = output_path / topic_dir.name / lang_dir.name
            os.makedirs(output_dir, exist_ok=True)
    
            files = list(lang_dir.glob('*'))
            for filename in tqdm(files, desc=f'{lang_dir}'):
                file_id, lang, _, _, _, text = dataset.load_raw(filename)
                
                preds, lemmas, words = get_token_info(nlp(text))
                if len(preds) != len(lemmas) != len(words):
                    raise ValueError("The counts do not match")

                df = pd.DataFrame(columns=['words', 'lemmas', 'preds'])
                df['words'] = words
                df['lemmas'] = lemmas
                df['preds'] = preds
                write_predictions(df, lang, topic, file_id, output_dir)


if __name__ == '__main__':
    typer.run(main)
