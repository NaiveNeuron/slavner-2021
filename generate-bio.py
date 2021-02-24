from ast import literal_eval
import pandas as pd
import typer
from pathlib import Path
from tqdm import tqdm


def write_bio(df: pd.DataFrame,
              output_column: str,
              output_path: Path,
              separator: str = '\t'):
    with output_path.open('w') as f:
        itertuple = zip(df['words'], df[output_column])
        for words, outputs in tqdm(itertuple,
                                   total=len(df),
                                   desc=f"{output_path}"):

            # Covert the strings saved to .csv into Python lists
            words = literal_eval(words)
            outputs = literal_eval(outputs)

            for word, out in zip(words, outputs):
                f.write(word + separator + out + '\n')
            f.write('\n')


def df_to_bio(df: pd.DataFrame,
              output_column: str,
              output_path: Path,
              separator: str = '\t',
              split_by: str = None):
    if split_by is not None:
        splits = df.groupby(split_by)
    else:
        splits = [("", df)]

    for split_name, df_split in splits:
        new_filename = output_path.stem + f"_{split_name}" + output_path.suffix
        split_path = output_path.parent / new_filename
        write_bio(df_split, output_column, split_path, separator)


def main(
    preprocessed_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True
    ),
    train_path: Path = typer.Argument(
        ...,
        dir_okay=False,
        writable=True
    ),
    valid_path: Path = typer.Argument(
        ...,
        dir_okay=False,
        writable=True
    ),
    validation_topic: str = typer.Option(...),
    output_column: str = typer.Option(...),
    delimiter: str = typer.Option('\t'),
    split_by: str = typer.Option(None),
):
    df = pd.read_csv(preprocessed_path)

    df_train = df[df['topic'] != validation_topic]
    df_val = df[df['topic'] == validation_topic]

    typer.echo(f"Writing train data to {train_path}")
    df_to_bio(df_train, output_column, train_path, separator=delimiter, split_by=split_by)

    typer.echo(f"Writing valid data to {valid_path}")
    df_to_bio(df_val, output_column, valid_path, separator=delimiter, split_by=split_by)


if __name__ == '__main__':
    typer.run(main)
