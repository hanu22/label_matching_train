from ast import literal_eval
from pathlib import Path

import pandas as pd


def read_processed_file(file_path: Path) -> pd.DataFrame:
    """Reads file as Pandas dataframe.

    Convert string list objects into literal list object
    Convert labels to uppercase (this cast it into a string > define type again)

    Args:
        file_path (Path): Location of file in working directory

    Returns:
        pd.DataFrame: dataframe
    """
    try:
        df = (
            pd.read_csv(
                file_path,
                usecols=[
                    "text",
                    "label",
                ],
                dtype={
                    "text": str,
                    "label": "category",
                },
                converters={
                    "string_list": literal_eval,
                },
            )
            .assign(
                label=lambda df: df["label"].str.upper(),
            )
            .set_index("pvid")
            .astype(
                {
                    "label": "category",
                }
            )
        )

    except FileNotFoundError as e:
        print("The exception: ", e)

    else:
        return df
