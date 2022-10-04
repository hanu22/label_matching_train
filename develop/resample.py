import pandas as pd


def downsample(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Downsamples a dataframe to have equal number of labels.

    Args:
        df (pd.DataFrame): dataframe to downsample

    Returns:
        pd.DataFrame: downsampled dataframe
    """
    # Get the number of samples per label > get the min number of samples
    min_label_count = min(df["label"].value_counts())

    # resample by the the min number of samples
    df = df.groupby(["label"]).apply(
        lambda c: c.sample(
            n=min_label_count,
            replace=False,
            random_state=42,
        ),
    )

    # Drop level from Multi-Index
    df.index = df.index.droplevel(0)

    return df


def replace_labels(
    df: pd.DataFrame,
    to_replace: list[str],
    value: str,
) -> pd.DataFrame:
    """Replaces a list of labels with a new label.

    Args:
        df (pd.DataFrame): dataframe to process
        to_replace (List[str]): list of labels to replace
        value (str): new label

    Returns:
        pd.DataFrame: processed dataframe
    """
    df["label"] = df["label"].replace(
        to_replace=to_replace,
        value=value,
    )

    return df


def sample_by_count(
    df: pd.DataFrame,
    count: int,
) -> pd.DataFrame:
    """Samples a dataframe by given count.

    Args:
        df (pd.DataFrame): dataframe to sampled
        count (int): number of samples required

    Returns:
        pd.DataFrame: sampled dataframe
    """
    # Get labels with samples > count
    # Keep only True
    labels = df["label"].value_counts() > count
    labels = labels[labels].index

    # Resample samples with those labels by count
    # Do not replace
    df_higher = (
        df.query("label in @labels")
        .groupby(by="label")
        .apply(
            lambda c: c.sample(
                n=count,
                replace=False,
                random_state=42,
            )
        )
    )

    df_higher.index = df_higher.index.droplevel(0)

    # Get labels with samples <= count
    # Keep only True
    labels = df["label"].value_counts() <= count
    labels = labels[labels].index

    # Get samples with those labels
    df_lower = df.query("label in @labels")

    # Join dfs with balanced samples
    df_balanced = df_higher.append(df_lower).reset_index(drop=True)

    #  get excluded samples
    # df_excluded = df[~df["text"].isin(df_balanced["text"])].dropna(
    #     subset=["text", "label"]
    # )

    return df_balanced
