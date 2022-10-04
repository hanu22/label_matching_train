from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("white")
sns.set_palette("Set2")


# TODO: can we plot NERs distinctiveness?


def plot_samples_per_market(
    df: pd.DataFrame,
    n_markets: int,
):
    """Plots number of samples per market.

    Args:
        df (pd.DataFrame): dataframe
        n_markets (int): number of markets to plot
    """
    fig, ax = plt.subplots(
        figsize=(6, 6),
        tight_layout=True,
    )

    # Number of samples per market for the top tm_no
    df["market"].value_counts().head(n_markets).plot(kind="bar", ax=ax)

    ax.set(
        title="Number of samples per market",
        xlabel="Market",
        ylabel="Samples No.",
    )

    # Save figure
    fig.savefig(
        Path(
            "develop",
            "eda",
            "samples_per_market.png",
        )
    )


def plot_samples_per_market_hue(
    df: pd.DataFrame,
    n_markets: int,
    hue: str,
):
    """Plots number of samples per market.

    Args:
        df (pd.DataFrame): dataframe
        n_markets (int): number of TM to plot
        hue (str): column for hue
    """
    fig, ax = plt.subplots(
        figsize=(6, 6),
        tight_layout=True,
    )

    # Number of samples per market for the top 10 per category
    sns.countplot(
        x="market",
        hue=hue,
        data=df,
        order=df["market"].value_counts().head(n_markets).index.to_list(),
        ax=ax,
    )

    ax.set(
        title="Number of samples per market",
        xlabel="Market",
        ylabel="Samples No.",
    )

    # Save figure
    fig.savefig(
        Path(
            "develop",
            "eda",
            "samples_per_market_hue.png",
        )
    )


def count_ners(s: pd.Series) -> int:
    """Counts entities in a column of lists.

    Some cells have ["None"] > replace with np.nan > drop

    Args:
        s (pd.Series): column with entities

    Returns:
        int: sum of entities
    """
    return (
        s.map(lambda cell: np.nan if cell == ["None"] else cell)
        .dropna()
        .apply(len)
        .sum()
    )


def plot_ner_count(
    df: pd.DataFrame,
    cols: list[str],
):
    """Plots count of ground truth entities, per type.

    Args:
        df (pd.DataFrame): dataframe with ground truth entities
        cols (list[str]): list of NER columns
    """
    # Group by market ->
    # count NERs per column per market ->
    # sum NERs per column ->
    # pd.Series
    s = df.groupby("market").agg(count_ners).sum()

    # Select only NER rows in series
    s = s.loc[cols].sort_values(ascending=False)

    fig, ax = plt.subplots(
        tight_layout=True,
    )

    # Count of all entities
    sns.barplot(
        x=s.index,
        y=s.values,
        ax=ax,
    )

    ax.set(
        title="Count of entities per type",
        xlabel="Samples No.",
        ylabel="",
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    # Save figure
    fig.savefig(
        Path(
            "develop",
            "eda",
            "ents_per_type.png",
        )
    )


def plot_ner_count_per_market(
    df: pd.DataFrame,
    cols: list[str],
):
    """Plots count of ground truth entities, per market, per type.

    Args:
        df (pd.DataFrame): dataframe with ground truth entities
        cols (list[str]): list of NER columns
    """
    # Group by market ->
    # count NERs per column per market ->
    # sum NERs per column ->
    # pd.Series
    s = df.groupby("market").agg(count_ners)

    # Select only NER rows in series
    s = s[cols]

    # Convert data to long format
    s = (
        s.melt(ignore_index=False)
        .reset_index()
        .sort_values(
            by=["market", "value"],
            ascending=False,
        )
    )

    fig, ax = plt.subplots(
        tight_layout=True,
    )

    sns.barplot(
        x="value",
        y="market",
        hue="variable",
        data=s,
        ax=ax,
    )

    ax.set(
        title="Count of entities per market and type",
        xlabel="Samples No.",
        ylabel="",
    )

    fig.savefig(
        Path(
            "develop",
            "eda",
            "ents_per_type_per_market.png",
        )
    )
