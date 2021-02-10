import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import logging
logging.basicConfig()


def load_annotated(f_annotated: Path):
    text = f_annotated.read_text()
    lines = text.strip().split('\n')
    file_id = lines[0]

    annotations = {}

    for i, line in enumerate(lines[1:]):
        split = line.split('\t')

        if len(split) != 4:
            logging.warning('File %s: Line %d "%s" is split into %d columns',
                            file_id, i + 1, line, len(split))
            continue

        token, lemma, tag, entity = split

        match = {
            "token": token,
            "lemma": lemma,
            "tag": tag,
            "entity": entity
        }

        existing = annotations.get(token)
        # If a match already exists, there is a chance something is messed up
        if existing:
            logging.warning('File %s: Line %d "%s" contains seen token.',
                            file_id, i + 1, line)

            # If the new match is not the same as it was before, there is a
            # chance at least some part of the labeling is wrong.
            if existing != match:
                logging.warning('File %s: Line %d, Ours: %r Existing: %r',
                                file_id, i + 1, match, existing)

            continue

        annotations[token] = match
    return file_id, annotations


def load_raw(f_raw: Path):
    text = f_raw.read_text()
    lines = text.strip().split('\n')

    txt_id, language, creation_date, url, title = lines[:5]

    text = ' '.join(lines[6:])

    return txt_id, language, creation_date, url, title, text


def process_file(f_annotated: Path, f_raw: Path):
    return


if __name__ == '__main__':
    for f in Path('./bsnlp2021_train_r1/annotated/').glob('**/*.out'):
        load_annotated(f)

    # for f in Path('./bsnlp2021_train_r1/raw/').glob('**/*.txt'):
    #     r = load_raw(f)
