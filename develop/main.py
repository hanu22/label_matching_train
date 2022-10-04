# %%
import logging
from pathlib import Path

import pandas as pd
from load import read_raw_file
from preprocess import filter_markets, filter_similar_texts, sanitize
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
    handlers=[RichHandler(markup=True)],
)
logger = logging.getLogger(__name__)

# %%
# Load raw data
df = read_raw_file(
    Path(
        "data",
        "raw",
        "",
    )
)

df.info()

# %%
# Preprocess data

df = filter_markets(
    df=df,
    market_col="market",
)

df = sanitize(
    df,
    cols=[],
)

# Remove similar texts for each market
markets = df["market"].unique()

df_markets_keep = []
df_markets_remove = []

for market in markets:
    print(market.replace(",", ""))

    df_market = df.query("market == @market")

    # Create dataframe from list of found indexes
    df_keep, df_remove = filter_similar_texts(
        df_market,
        identifier=market.replace(",", ""),
    )

    df_markets_keep.append(df_keep)
    df_markets_remove.append(df_remove)

df_keep = pd.concat(df_markets_keep)
df_remove = pd.concat(df_markets_remove)

df.info()

# %%
# EDA

# %%
# Resampling

# Classification

# Check labels frequency and resample
df["label"].value_counts()

# Get unique labels
labels = df["label"].unique()

# For multilabel classification, each 'text' might have more than one label
# Create a set of labels for each example
df = df.groupby("text")["label"].apply(set).reset_index()

# For binary classification, the created set of labels should only have 1 label
# Resolve/drop rows with more than 1 label
len(df[df["label"].apply(len) > 1])
df = df[df["label"].apply(len) == 1]

# Entities

# %%
# Translate

# %%
# Save processed data to processed
df.to_csv(
    Path(
        "data",
        "processed",
        "",
    ),
    columns=[
        "text_orig",
        "text_translated",
        "text",
        "label",
    ],
    index=True,
)

# %%
