# %%
from collections import Counter
from pathlib import Path
from typing import List

import pandas as pd
from prodigy.components.db import connect

# %%
# connect to local sqlite database
# change path as appropriate

db = connect(
    "sqlite",
    {
        "name": "prodigy.db",
        "path": Path("prodigy", "local"),
    },
)

# %%
# connect to Azure MySQL database

db = connect(
    db_id="mysql",
    db_settings={
        "user": "prodigy@data-science-automation-dev-eun-mysql",
        "password": "eVV66177Zb",
        "host": "data-science-automation-dev-eun-mysql.mysql.database.azure.com",
        "port": 3306,
        "database": "prodigy",
        "ssl": {"ssl": {"ssl-ca": "/var/www/html/BaltimoreCyberTrustRoot.crt.pem"}},
    },
)

# %%
datasets_list = db.datasets
sessions_list = db.sessions

db.reconnect()

meta = db.get_meta("xxx")
dataset = db.get_dataset("training_5000")

# %%
# read dataset into dataFrame
df = pd.DataFrame(dataset)

# %%
# count samples number per annotator
df.groupby("_session_id").count()["text"]

# %%
# count samples number per label for binary labels
df.groupby("label").count()["text"]

# %%
# count samples number per label for multi-labels
Counter(item[0] for item in df["accept"] if item)


# %%
# count samples number per NER
count = Counter()


def count_ner(c):
    for item in c:
        count[item] += 1


df.apply(
    lambda s: list(
        map(
            lambda item: item["label"],
            s.spans,
        )
    ),
    axis=1,
).map(count_ner)

print(count)


# %%
def select_ner(spans: List[dict]) -> List[dict]:
    """Only select specific NERs in spans.

    Args:
        spans (List[dict]): list of spans, each containing the label.
        An example without any spans is an empty list

    Returns:
        List[dict]: filtered list of spans
    """
    target_ners = ["MAIN_ING", "QUID"]
    filtered_spans = []

    for span in spans:
        if span["label"] in target_ners:
            filtered_spans.append(span)

    return filtered_spans


# select only target NERs
# remove rows where the spans list is empty
df["spans"] = df["spans"].apply(select_ner)
df = df[df["spans"].str.len() != 0]
count_ner()

# %%
# save dataset into jsonl file
df.to_json(
    Path(
        "..",
        "data",
        f"dataset_annotated_{len(dataset)}.jsonl",
    ),
    orient="records",
    force_ascii=True,
    lines=True,
)

# %%
# db.drop_dataset('xxx')
db.close()

# %%
