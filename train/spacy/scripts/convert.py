import pandas as pd
import spacy
from spacy.tokens import Doc

# A blank pipeline loads just a tokenizer
# Sentencizer is needed (otherwise, we get Sentence boundaries unset error)
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

# Use lowercase token text for the internal phrase matcher
# Both the patterns and the text will be case-insensitive matched
config = {
    "phrase_matcher_attr": "LOWER",
    "validate": True,
}


def convert_cats_to_doc(
    text: str,
    text_labels: set[str],
    labels: list[str],
) -> Doc:
    """Makes a doc from text with labels.

    We already have labels -> use them to label a text in doc.cats

    Args:
        text (str): raw text
        text_labels set[str]: text labels
        labels (list[str]): list of unique labels

    Returns:
        Doc: a spaCy doc object
    """
    # docs_to_json expects a dict of cats such as
    # {"cat1": 1.0, "cat2": 0.0, ...}
    cats = {}

    for label in labels:
        if label in text_labels:
            cats[label] = 1.0
        else:
            cats[label] = 0.0

    # Make a doc object and update its cats
    doc = nlp(text)
    doc.cats = cats

    return doc


def convert_ners_to_doc(
    row_gt: pd.Series,
    text: str,
) -> Doc:
    """Makes a doc from a number of ground truth NERs.

    We already have NER labels -> use them to label a text in doc.ents

    Args:
        row_gt (pd.Series): row of ground truth
        text (str): raw text

    Returns:
        Doc: a spaCy doc object
    """
    # If the entity ruler exists > replace it each time,
    # so patterns are not carried over from previous rows
    if nlp.has_pipe(name="entity_ruler"):
        ruler = nlp.replace_pipe(
            name="entity_ruler",
            factory_name="entity_ruler",
            config=config,
        )
    else:
        # Initialise the entity ruler first time
        ruler = nlp.add_pipe(
            name="entity_ruler",
            factory_name="entity_ruler",
            config=config,
        )

    patterns = []

    # Make a list of patterns in the format: {"label": "", "pattern": ""}
    # Each ind_values is a list
    # Add id as the ind_name so we can use it to remove the pattern
    for ind_name, ind_values in row_gt.items():
        for col_value in ind_values:
            patterns.append(
                {
                    "label": ind_name.upper(),
                    "pattern": col_value,
                    "id": ind_name,
                }
            )

    ruler.add_patterns(patterns)

    # Make a doc object that will includes NERs found by the entity ruler
    doc = nlp(text)

    return doc


def convert_ners_to_spancat_doc(
    doc: Doc,
    span_cats: list[str],
) -> Doc:
    """Takes row of ground truth and adds them as entities.

    We already have a doc with NERs labels -> convert some to spanCat in doc.spans

    Args:
        doc (Doc): spacy doc object
        span_cats (list[str]): list of spancat labels to find in lowercase

    Returns:
        Doc: a spaCy doc object
    """
    if span_cats:

        # Create an empty list to add found SpanCat entities
        ents_spancat = []

        # Create an empty list to keep non-SpanCat entities
        # We can't remove entities directly from doc.ents as it's a Tuple (immutable)
        ents_remaining = []

        # Loop over all entities > split into SpanCat and non-SpanCat entities
        # Existing labels are uppercase -> lowercase
        for ent in doc.ents:
            if ent.label_.lower() in span_cats:
                ents_spancat.append(ent)
            else:
                ents_remaining.append(ent)

        # Reassign the remaining entities back as a list
        doc.ents = ents_remaining

        # Assign found SpanCat entities as a list
        doc.spans["sc"] = ents_spancat

    return doc
