# %%
import logging
from pathlib import Path

from convert import (
    convert_cats_to_doc,
    convert_ners_to_doc,
    convert_ners_to_spancat_doc,
)
from load import read_processed_file
from rich.logging import RichHandler
from split import split_train_dev

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
    handlers=[RichHandler(markup=True)],
)

# %%
# Load processed data
df = read_processed_file(
    Path(
        "data",
        "processed",
        "",
    )
)

df.inf()

# %% Classification

# Get unique labels
labels = df["label"].unique()

df["doc"] = df.apply(
    lambda s: convert_cats_to_doc(
        text=s["text"],
        label=s["label"],
        labels=labels,
    ),
    axis=1,
)

# %% Entities

df["doc"] = df.apply(
    lambda s: convert_ners_to_doc(
        row_gt=s.loc[[]],
        text=s.loc["text"],
    ),
    axis="columns",
)

# Convert few entities from NERs to SpanCat
df["doc"] = df.apply(
    lambda s: convert_ners_to_spancat_doc(
        doc=s["doc"],
        span_cats=[],
    ),
    axis="columns",
)

# %%
# Create spacy and JSON files
split_train_dev(
    df,
    stratify_by=None,
)

# %%
