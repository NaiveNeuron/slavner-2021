from pathlib import Path
from typing import List, Tuple
from dataset import SlavNERDataset
from enum import Enum
import typer


class Tokenizers(str, Enum):
    nltk = "nltk"


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


def main(
    output_path: Path = typer.Argument(
        ...,
        exists=False,
        dir_okay=False,
        readable=True
    ),
    tokenizer: Tokenizers = typer.Option(
        Tokenizers.nltk,
        case_sensitive=False
    ),
):
    if tokenizer == 'nltk':
        word_tokenizer, sent_tokenizer = get_nltk_tokenizers()

    dataset = SlavNERDataset(
        data_dir=Path('./bsnlp2021_train_r1/'),
        word_tokenizer=word_tokenizer,
        sent_tokenizer=sent_tokenizer,
    )
    dataset.to_df().to_csv(output_path, index=False)


if __name__ == '__main__':
    typer.run(main)
