from pathlib import Path
from random import sample
from typing import Union

import joblib
import numpy as np
from annoy import AnnoyIndex
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Create here so it can be used in any function
# Convert strings to lower cased tf-idf vectors
# Do not remove stop words as this should be language agnostic
# Match any word with [a-z] with at least 2 chars long
# Regex does not work for accents -> strip it first
vectorizer = TfidfVectorizer(
    lowercase=True,
    analyzer="word",
    strip_accents="unicode",
    token_pattern=r"\b[a-z]{2,}\b",
)


def reduce_dims_pca(
    dense_matrix: np.ndarray,
    scale: bool,
    identifier: str,
) -> np.ndarray:
    """Reduces matrix dimensions using PCA.

    Number of components should preserve 90% of the variance ratio

    Sklearn PCA class:
    - Centers each feature (i.e. mean=0)
    - Does not scale features -> requires scaling
    - Does not support sparse input

    Args:
        dense_matrix (np.ndarray): dense matrix
        scale (bool): use StandardScaler if true
        identifier (str): to use for saving .pkl to disk

    Returns:
        np.ndarray: Transformed dense matrix. Shape: len(texts) x len(components_)
    """
    features_no = dense_matrix.shape[1]

    saved_pca_path = Path("develop", "artifacts", f"{identifier}_pca.pkl")

    # Try loading PCA
    try:
        print(f"Loading {saved_pca_path}...")
        pca = joblib.load(saved_pca_path)

    # If it doesn't exist -> catch error
    # Set trained features numbers to 0 so we can later use it
    except FileNotFoundError:
        print("File doesn't exist.")
        trained_features_no = 0
        pass

    else:
        # Get the number of input features
        # Note this is different than the number of output features
        trained_features_no = pca.n_features_in_

    finally:

        # True if PCA failed to load OR loaded but different number of features
        if trained_features_no != features_no:
            print("Inconsistent dimensions. Training a new PCA...")

            pca = PCA(
                n_components=0.90,
                random_state=42,
            )

            if scale:
                scaler = StandardScaler()
                dense_matrix = scaler.fit_transform(dense_matrix)

            # PCA doesn't support sparse data
            pca.fit(dense_matrix)

            # Save after fitting
            joblib.dump(pca, saved_pca_path)

    print(f"Number of components: {len(pca.components_)}")

    reduced_dense_matrix = pca.transform(dense_matrix)

    return reduced_dense_matrix


def create_tfidf_matrix(
    texts: list[str],
    dense: bool,
) -> Union[csr_matrix, np.ndarray]:
    """Creates a TFIDF matrix from list of texts.

    We need a dense matrix for using df.apply(). Otherwise, we get:
    Numpy TypeError: ufunc 'isfinite' error

    Args:
        texts (list[str]): list of texts
        dense (bool): return dense matrix if true

    Returns:
        Union[csr_matrix, np.ndarray]: Compressed sparse row matrix or dense matrix
        Shape: len(texts) x len(vocabulary)
    """
    # Fit and transform all texts into tf-idf sparse matrix
    # Output is a sparse csr (Compressed Sparse Row) matrix
    sparse_tfidf_matrix = vectorizer.fit_transform(texts)

    # Debug only
    # vectorizer.get_feature_names()

    # Get the number of features used in the vocabulary
    num_terms = len(vectorizer.get_feature_names_out())

    print(f"Number of features: {num_terms}")

    # Don't use todense() as it returns np.matrix which gets a deprecation warning
    # Use toarray() instead as it returns ndarray
    if dense:
        dense_tfidf_matrix = sparse_tfidf_matrix.toarray()
        return dense_tfidf_matrix

    else:
        return sparse_tfidf_matrix


def find_nns(
    ann_index: AnnoyIndex,
    n: int = 50,
) -> list[tuple[list[int], list[float]]]:
    """Finds n nearest neighbors and their distances.

    Args:
        ann_index (AnnoyIndex): annoy index
        n (int, optional): number of approximate nearest neighbors. Defaults to 50.

    Returns:
        list[tuple[list[int], list[float]]]: list of nns and their distances
    """
    # List to store n nearest neighbors for each item and their distances
    # Each item in the list is a tuple of:
    # list[n nearest neighbors], list[n nearest neighbors distances]
    items_nns = []

    # Get number of items
    n_items = ann_index.get_n_items()

    # Loop over all items from 0 to n_items -1
    for item_ind in range(n_items):

        # For each item, get nns and their distances
        # Distances are sorted in an ascending order (highest first)
        nns_by_item = ann_index.get_nns_by_item(
            i=item_ind,
            n=n,
            include_distances=True,
        )

        # Append as a tuple for consistency
        items_nns.append(nns_by_item)

    return items_nns


def filter_nns(
    items_nns: list[list[int], list[float]],
    threshold: float,
) -> list[list[int]]:
    """Filters nns by threshold distance.

    Args:
        items_nns (list[list[int], list[float]]): n nearest neighbors and distances
        threshold (float): threshold to use against distance

    Returns:
        list[list[int]]: filtered n nearest neighbors for each item
    """
    # List to store filtered n nearest neighbors for each item
    # Each item in the list is a tuple of list[n nearest neighbors]
    filtered_items_nns = []

    # Loop over each item_nns -> unpack
    for nns_indices, distances in items_nns:

        # List to store current item filtered nns
        filtered_item_nns = []

        # Loop over corresponding pairs of index, distance
        for nn_ind, distance in zip(nns_indices, distances):

            # distance is within [0, 2] ->
            # only select nns indices with distance less than threshold
            if distance < threshold:
                filtered_item_nns.append(nn_ind)

        # Append as a sorted list
        # We sort so lists with similar items are identical
        filtered_items_nns.append(sorted(filtered_item_nns))

    return filtered_items_nns


def get_similar_text_indices_annoy(
    tfidf_matrix: Union[csr_matrix, np.ndarray],
    threshold: float,
    identifier: str,
) -> list[tuple[list[int], list[float]]]:
    """Uses annoy to find indices of similar texts.

    WARNING: the `dot` metric doesn't work with PCA projection!

    `angular` is equivalent to `euclidean` of normalized vectors -> within [0, 2]
    See Issue #558

    Args:
        tfidf_matrix (Union[csr_matrix, np.ndarray]): tfidf matrix
        threshold (float): threshold to use against distance
        identifier (str): to use for saving .ann to disk

    Returns:
        list[tuple[list[int], list[float]]]: list of tuples: list(index), list(similarities)
    """
    # Get number of features (i.e. unique vocabulary)
    features_no = tfidf_matrix.shape[1]

    # Initialize annoy index with metric
    ann_index = AnnoyIndex(
        f=features_no,
        metric="angular",
    )

    # Add matrix to Annoy, item by item
    # Each item is one text
    for item_ind, item_vector in enumerate(tfidf_matrix):

        ann_index.add_item(
            item_ind,
            item_vector,
        )

    saved_index_path = Path("develop", "artifacts", f"{identifier}_annoy.ann")

    # Try loading existing index on disk
    try:
        print(f"Loading {saved_index_path}...")
        ann_index.load(str(saved_index_path))

    # If the index doesn't exists or vector dimensions are different to existing one ->
    # we will get an error -> catch -> build and save a new index
    except OSError:

        print("File doesn't exist or inconsistent dimensions. Building a new index...")

        # Build a forest of n_trees trees
        # More trees gives higher precision when querying
        # Use all CPUs available
        ann_index.build(
            n_trees=500,
            n_jobs=-1,
        )

        # Save the index to disk
        ann_index.save(str(saved_index_path))

    # Get list of nns for all items
    items_nns = find_nns(ann_index)

    # Get list of filtered nns for each item
    # Each list is sorted
    filtered_items_nns_indices = filter_nns(items_nns, threshold)

    return filtered_items_nns_indices


def get_nns_indices_to_remove(
    nns_indices: list[list[int]],
) -> list[int]:
    """Gets a list of nns indices to remove.

    Args:
        nns_indices (list[list[int]]): nns indices to process

    Returns:
        list[int]: nns indices to remove
    """
    nns_indices_to_remove = []

    # Set only works on hashable types ->
    # convert from list of lists to list of tuples ->
    # apply set() ->
    # create list of unique lists
    for sublist in set(map(tuple, nns_indices)):

        # Sample a number of items = length of list -1
        # TODO: keep the item with highest occurrence
        random_nns_indices = sample(sublist, len(sublist) - 1)

        # Lists with 1 item only will result in empty sampled list
        if random_nns_indices:
            nns_indices_to_remove.extend(random_nns_indices)

    # Do not sort so similar NNs stay together
    # Remove duplicates
    # TODO: investigate why there are duplicates
    unique_nns_indices_to_remove = list(set(nns_indices_to_remove))

    return unique_nns_indices_to_remove
