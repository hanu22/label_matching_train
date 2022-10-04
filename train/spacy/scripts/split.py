import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from spacy.tokens import DocBin


def save_json(
    docs: pd.Series,
    split: str,
) -> json:
    """Saves column of doc objects into JSON file.

    Args:
        docs (pd.Series): column with doc objects
        split (str): `train` or `dev`

    Returns:
        json: file.json
    """
    # Convert doc object into a JSON format
    # Add default_handler, otherwise, we get OverflowError
    # https://pandas.pydata.org/pandas-docs/version/0.13.0/io.html#fallback-behavior
    docs_json = docs.apply(lambda doc: doc.to_json())

    # Save json object as a JSON file
    docs_json.to_json(
        Path(
            "train",
            "spacy",
            "assets",
            f"{split}.json",
        ),
        default_handler=str,
        orient="records",
    )


def save_doc_bin(
    docs: pd.Series,
    split: str,
) -> DocBin:
    """Saves column of doc objects into DocBin binary file.

    Args:
        docs (pd.Series): column with doc objects
        split (str): `train` or `dev`

    Returns:
        DocBin: file.spacy
    """
    # No need to covert to list of docs
    db = DocBin(docs=docs)

    db.to_disk(
        str(
            Path(
                "train",
                "spacy",
                "corpus",
                f"{split}.spacy",
            )
        )
    )


def split_train_dev(
    df: pd.DataFrame,
    stratify_by: pd.Series,
) -> None:
    """Splits data in column and saves into binary and json files.

    Args:
        df (pd.DataFrame): dataframe with `doc` column to split
        stratify_by (pd.Series): column to stratify by
    """
    # Returns pd.series
    train_docs, dev_docs = train_test_split(
        df["doc"],
        test_size=0.3,
        stratify=stratify_by,
        random_state=42,
        shuffle=True,
    )

    save_doc_bin(train_docs, "train")
    save_json(train_docs, "train")

    save_doc_bin(dev_docs, "dev")
    save_json(dev_docs, "dev")
