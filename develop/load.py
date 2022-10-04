from functools import lru_cache
from pathlib import Path

import pandas as pd


@lru_cache
def read_raw_file(file_path: Path) -> pd.DataFrame:
    """Reads file as Pandas dataframe.

    Sort by 'frequency', so the text with the highest frequency is on top
    This would be useful when sampling

    Make a copy of 'text' into 'text_orig' for reference
    Make a new None column 'text_translated' for later translation

    Args:
        file_path (Path): file path

    Returns:
        pd.DataFrame: dataframe
    """
    try:
        df = (
            pd.read_csv(
                file_path,
                usecols=[],
                dtype={
                    "text_col": str,
                    "label_col": "category",
                },
                na_values=["", " "],
                converters={},
            )
            .rename(
                columns={
                    "text_col": "text",
                    "label_col": "label",
                }
            )
            .dropna(
                subset=[
                    "text",
                ]
            )
            .fillna("None")
            .assign(
                text_orig=lambda df: df["text"],
            )
            .sort_values(
                by="frequency",
                ascending=False,
            )
            .set_index("pvid")
        )

    except FileNotFoundError as e:
        print("The exception: ", e)

    else:
        return df
