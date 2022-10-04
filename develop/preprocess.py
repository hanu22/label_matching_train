import numpy as np
import pandas as pd
from similarity import (
    create_tfidf_matrix,
    get_nns_indices_to_remove,
    get_similar_text_indices_annoy,
    reduce_dims_pca,
)

markets_english = ["AU", "GB,IE", "US"]
markets_non_english = ["BE", "CZ", "HU", "NL", "PL", "SK", "IT", "RO"]
markets = markets_english + markets_non_english


def filter_markets(df: pd.DataFrame) -> pd.DataFrame:
    """Filters markets in levels 1, 2, and 3 only.

    For reference:
    https://brandbank.atlassian.net/wiki/spaces/NBB/pages/2960162828/Target+Markets

    Args:
        df (pd.DataFrame): dataframe

    Returns:
        pd.DataFrame: filtered dataframe
    """
    # Map market combinations
    df["market"] = df["market"].replace(
        {
            "BE,LU": "BE",
            "GB": "GB,IE",
            "NL, LU": "NL",
            "IE": "GB,IE",
        }
    )

    return df.loc[df["market"].isin(markets)]


def sanitize(
    df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """Sanitizes data after initial loading.

    Remove leading, trailing, or 2+ spaces from specified columns
    Replace empty string "" with nan
    Drop specified columns rows with any missing values
    # Drop duplicated specified columns

    Args:
        df (pd.DataFrame): dataframe
        cols (list[str]) : list of columns to sanitize

    Returns:
        pd.DataFrame: dataframe
    """
    # Remove leading, trailing, or 2+ spaces
    # Replace empty string "" with nan
    # from specified columns
    # Columns with nan will be converted to float
    df[cols] = (
        df[cols]
        .applymap(lambda cell: cell.strip())
        .replace(
            to_replace=r"\s+",
            value=" ",
            regex=False,
        )
        .replace(
            to_replace="",
            value=np.nan,
            regex=False,
        )
    )

    # Drop specified columns rows with any missing values
    # Drop duplicated specified columns resulting from sanitization
    # Convert float columns back to str
    df = (
        df.dropna(subset=cols)
        .drop_duplicates(subset=cols)
        .astype({col: str for col in cols})
    )

    return df


def split_string(
    df: pd.DataFrame,
    cols: str,
    split_on: str,
) -> pd.DataFrame:
    """Splits strings in columns that are concatenated with an operator (e.g. "| ").

    Args:
        df (pd.DataFrame): dataframe
        cols (list[str]): list of columns to split on operator
        split_on (str): chars to split on

    Returns:
        pd.DataFrame: dataframe
    """
    return df[cols].applymap(lambda cell: cell.split(split_on))


def filter_similar_texts(
    df: pd.DataFrame,
    identifier: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filters dataframe from redundant texts.

    Args:
        df (pd.DataFrame): dataframe to filter
        identifier (str): for saving PCA and Annoy files (e.g. market)

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: dataframes to keep and remove
    """
    # Vectorise texts into TFIDF
    dense_tfidf_matrix = create_tfidf_matrix(
        texts=df["text"].to_list(),
        dense=True,
    )

    # TFIDF results in a large number of features -> use pca to reduce dimensions
    # TFIDF is already normalized -> no need for scaling
    reduced_dense_tfidf_matrix = reduce_dims_pca(
        dense_matrix=dense_tfidf_matrix,
        scale=False,
        identifier=identifier,
    )

    # Use Annoy to find the nearest neighbors for each query text
    nns_indices = get_similar_text_indices_annoy(
        tfidf_matrix=reduced_dense_tfidf_matrix,
        threshold=0.1,
        identifier=identifier,
    )

    # Add to dataframe to inspect if needed
    df["nns_indices"] = nns_indices

    # Get nns indices to remove
    nns_indices_to_remove = get_nns_indices_to_remove(nns_indices)

    # Find sub-dataframe of texts to remove
    df_remove = df.iloc[nns_indices_to_remove]

    # Find sub-dataframe of remaining texts to keep
    df_keep = df.loc[~df.reset_index().index.isin(nns_indices_to_remove)]

    return df_keep, df_remove
