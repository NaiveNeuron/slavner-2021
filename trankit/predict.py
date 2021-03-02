from sacred import Experiment
from trankit import Pipeline, verify_customized_pipeline
from pathlib import Path
from typing import Dict
from collections import Counter
import re

ex = Experiment()


@ex.config
def config():
    category = 'customized-ner' # noqa
    lang = None # noqa
    save_dir = './save_dir' # noqa
    raw_data_dir = None # noqa
    output_data_dir = None # noqa


def symlink_pipeline_part(lang, save_dir, category, part):
    mapping = {
        'bg': 'bulgarian',
        'cs': 'czech',
        'pl': 'polish',
        'ru': 'russian',
        'sl': 'slovenian',
        'uk': 'ukrainian'
    }

    lang_prefix = str(Path('./langs') / mapping[lang] / mapping[lang])
    to_path = str(Path(save_dir) / category / category)

    p_to = Path(lang_prefix + part).resolve()
    p_from = Path(to_path + part)

    try:
        p_from.symlink_to(p_to)
    except FileExistsError:
        p_from.unlink()
        p_from.symlink_to(p_to)


def predict_on_text(pipeline: Pipeline, text: str) -> Dict:
    predictions = []
    r = pipeline(text)
    current_mwe = dict()
    for sentence in r['sentences']:
        for token in sentence['tokens']:
            token_ner = token['ner']
            if token_ner.startswith('S-'):
                predictions.append([
                    token['text'],
                    token['lemma'],
                    token_ner.replace('S-', '')
                ])
            elif token_ner.startswith('B-'):
                if 'text' in current_mwe:
                    current_mwe['text'] += ' ' + token['text']
                    current_mwe['lemma'] += ' ' + token['lemma']
                else:
                    current_mwe['text'] = token['text']
                    current_mwe['lemma'] = token['lemma']
            elif token_ner.startswith('I-'):
                current_mwe['text'] += ' ' + token['text']
                current_mwe['lemma'] += ' ' + token['lemma']
            elif token_ner.startswith('E-'):
                current_mwe['text'] += ' ' + token['text']
                current_mwe['lemma'] += ' ' + token['lemma']

                predictions.append([
                    current_mwe['text'],
                    current_mwe['lemma'],
                    token_ner.replace('E-', '')
                ])

                current_mwe = {}

    predictions_by_token = {}
    for prediction in predictions:
        text, lemma, tag = prediction
        if text not in predictions_by_token:
            predictions_by_token[text] = {
                'lemma': lemma,
                'tag': [tag]
            }
        else:
            predictions_by_token[text]['tag'].append(tag)

    return predictions_by_token


def generate_output_text(predictions: Dict) -> str:
    lines = []
    for token in sorted(predictions.keys(),
                        key=lambda x: x.lower()):
        prediction = predictions[token]
        tags = prediction['tag']
        lemma = prediction['lemma']

        counts = Counter(tags)
        # Get the first most common tag
        tag = counts.most_common(1)[0][0]

        # If we matched a token with both ORG and PER, return PER
        # http://bsnlp.cs.helsinki.fi/System_response_guidelines-1.2.pdf
        # (page 3)
        if 'ORG' in counts and 'PER' in counts:
            tag = 'PER'

        # If we matched a token with both ORG and PRO, return PRO
        # http://bsnlp.cs.helsinki.fi/System_response_guidelines-1.2.pdf
        # (page 3)
        elif 'ORG' in counts and 'PRO' in counts:
            tag = 'ORG'

        # If there is a dot as part of the token, ensure there is no whitespace
        # around it (like in 'W . Brytania' vs 'W.Brytania')
        if '.' in token:
            token = re.sub(r'\s+\.\s+', '.', token)
            lemma = re.sub(r'\s+\.\s+', '.', lemma)

        lines.append(f'{token}\t{lemma}\t{tag}\tORG-RAND')

    return '\n'.join(lines)


@ex.automain
def main(category, lang, save_dir, raw_data_dir, output_data_dir):

    symlink_pipeline_part(lang, save_dir, category, '.tagger.mdl')
    symlink_pipeline_part(lang, save_dir, category, '.vocabs.json')
    symlink_pipeline_part(lang, save_dir, category, '_lemmatizer.pt')
    symlink_pipeline_part(lang, save_dir, category, '.tokenizer.mdl')

    verify_customized_pipeline(
        category=category,
        save_dir=save_dir
    )

    p = Pipeline(lang=category,
                 cache_dir=save_dir)

    for file in Path(raw_data_dir).rglob('*'):
        print(file, file.name)
        file_text = file.read_text()
        lines = file_text.strip().split('\n')
        file_id = lines[0]
        text = ' '.join(lines[4:])
        predictions = predict_on_text(p, text)

        output_data_dir = Path(output_data_dir)
        # Ensure the output directories exist
        output_data_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_data_dir / file.name

        with output_path.open('w') as f:
            f.write(file_id + '\n')
            f.write(generate_output_text(predictions))
