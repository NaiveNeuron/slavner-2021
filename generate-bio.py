from ast import literal_eval
import pandas as pd
import typer
from pathlib import Path
from tqdm import tqdm


def df_to_bio(df: pd.DataFrame,
              output_column: str,
              output_path: Path,
              separator: str = '\t'):
    with output_path.open('w') as f:
        for i, row in tqdm(df.iterrows(),
                           total=len(df),
                           desc=f"{output_path}"):

            # Covert the strings saved to .csv into Python lists
            words = literal_eval(row['words'])
            outputs = literal_eval(row[output_column])

            for word, out in zip(words, outputs):
                f.write(word + separator + out + '\n')
            f.write('\n')


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
):
    df = pd.read_csv(preprocessed_path)

    df_train = df[df['topic'] != validation_topic]
    df_val = df[df['topic'] == validation_topic]

    typer.echo(f"Writing train data to {train_path}")
    df_to_bio(df_train, output_column, train_path)

    typer.echo(f"Writing valid data to {valid_path}")
    df_to_bio(df_val, output_column, valid_path)


if __name__ == '__main__':
    typer.run(main)
