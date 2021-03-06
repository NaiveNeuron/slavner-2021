import spacy
from sacred import Experiment
from pathlib import Path
from typing import Dict, Callable
from collections import Counter
import re

ex = Experiment()


@ex.config
def config():
    model_path = None # noqa
    lang =  None # noqa
    raw_data_dir = None # noqa
    output_data_dir = None # noqa


def predict_on_text(
    nlp: spacy.language.Language,
    lemmatizer: Callable,
    text: str
) -> Dict:

    docs = nlp(text)
    predictions = [(ent.text, ent.label_) for ent in docs.ents]

    predictions_by_token = {}
    for prediction in predictions:
        text, tag = prediction
        lemma = lemmatizer(text)
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


def generate_lemmatizer(lang: str):

    lang_to_model = {
        'ru': 'ru_core_news_lg',
        'pl': 'pl_core_news_lg'
    }

    nlp = None
    if lang in lang_to_model:
        nlp = spacy.load(lang_to_model[lang])

    def lemmatizer(text: str):
        if nlp is None:
            return text

        tokens = nlp(text)
        return ' '.join([token.lemma_ for token in tokens])

    return lemmatizer


@ex.automain
def main(model_path, lang, raw_data_dir, output_data_dir):

    nlp = spacy.load(model_path)
    lemmatizer = generate_lemmatizer(lang)

    for file in Path(raw_data_dir).rglob('*'):
        print(file, file.name)
        file_text = file.read_text()
        lines = file_text.strip().split('\n')
        file_id = lines[0]
        text = ' '.join(lines[4:])
        predictions = predict_on_text(nlp, lemmatizer, text)

        output_data_dir = Path(output_data_dir)
        # Ensure the output directories exist
        output_data_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_data_dir / file.name

        with output_path.open('w') as f:
            f.write(file_id + '\n')
            f.write(generate_output_text(predictions))
