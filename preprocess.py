from pathlib import Path
from typing import List, Tuple
from dataset import SlavNERDataset
from enum import Enum
import typer


class Tokenizers(str, Enum):
    nltk = "nltk"
    stanza = "stanza"


def get_nltk_tokenizers() -> Tuple:
    import nltk
    nltk.download('punkt')

    import nltk.tokenize
    word = nltk.tokenize.toktok.ToktokTokenizer()

    def word_tokenizer(text: str, lang: str) -> List[str]:
        return word.tokenize(text)

    def sent_tokenizer(text: str, lang: str = 'en') -> List[str]:
        lang_map = {
            'en': 'english',
            'sl': 'slovene',
            'cs': 'czech',
            'pl': 'polish',
            'uk': 'english',  # sadly, nltk does not have an Ukrainian model...
            'bg': 'english',  # sadly, nltk does not have a Bulgarian model...
            'ru': 'english'  # sadly, nltk does not have a Russian model...
        }
        return nltk.tokenize.sent_tokenize(text, lang_map[lang])
    
    return word_tokenizer, sent_tokenizer


def get_stanza_tokenizers() -> Tuple:
    import stanza

    cache_tokenizer = None 
    cache_lang = None

    def _get_tokenizer(lang: str):
        '''
        A simple "caching" mechanizsm as loding the tokenizer on each call is
        extremly expensive.
        '''
        nonlocal cache_tokenizer, cache_lang  # not pretty but works
        if cache_tokenizer is not None and lang == cache_lang:
            return cache_tokenizer
        stanza.download(lang)
        cache_tokenizer = stanza.Pipeline(lang=lang, processors='tokenize')
        cache_lang = lang
        return cache_tokenizer
        
    def word_tokenizer(text: str, lang: str = 'en') -> List[str]:
        tokenizer = _get_tokenizer(lang)
        doc = tokenizer(text)
        tokens = []
        for sent in doc.sentences:
            tokens.extend([token.text for token in sent.tokens if token.text])
        return tokens
    
    def sent_tokenizer(text: str, lang: str = 'en') -> List[str]:
        tokenizer = _get_tokenizer(lang)
        doc = tokenizer(text)
        return [sent.text for sent in doc.sentences]

    return word_tokenizer, sent_tokenizer


def main(
    output_path: Path = typer.Argument(
        ...,
        exists=False,
        dir_okay=False,
        readable=True
    ),
    input_path: Path = typer.Option(
        "./bsnlp2021_train_r1/",
        exists=True,
        dir_okay=True,
        file_okay=False
    ),
    tokenizer: Tokenizers = typer.Option(
        Tokenizers.nltk,
        case_sensitive=False
    ),
):
    if tokenizer == 'nltk':
        word_tokenizer, sent_tokenizer = get_nltk_tokenizers()
    elif tokenizer == 'stanza':
        word_tokenizer, sent_tokenizer = get_stanza_tokenizers()

    dataset = SlavNERDataset(
        data_dir=Path(input_path),
        word_tokenizer=word_tokenizer,
        sent_tokenizer=sent_tokenizer,
    )
    dataset.to_df().to_csv(output_path, index=False)


if __name__ == '__main__':
    typer.run(main)
