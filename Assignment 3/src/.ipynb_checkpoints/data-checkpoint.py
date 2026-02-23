"""
data.py - Data loading and preprocessing module for Goodreads genre classification.
"""

import gzip
import json
import pickle
import random
import requests
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

# ── URLs for the UCSD Goodreads dataset ──────────────────────────────────────
GENRE_URL_DICT = {
    "poetry":       "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic":"https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal":"https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "mystery_thriller_crime":"https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance":      "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult":  "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
    "history_biography":"https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
}


def load_reviews(url: str, head: int = 10000, sample_size: int = 2000) -> list:
    """Stream reviews from a gzipped JSON URL and return a random sample."""
    reviews = []
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with gzip.open(response.raw, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            reviews.append(d["review_text"])
            if head is not None and len(reviews) >= head:
                break
    if sample_size and len(reviews) > sample_size:
        reviews = random.sample(reviews, sample_size)
    return reviews


def build_genre_reviews_dict(
    genre_url_dict: dict = GENRE_URL_DICT,
    head: int = 10000,
    sample_size: int = 2000,
    cache_path: str = "genre_reviews_dict.pickle",
) -> dict:
    """Download (or load from cache) genre → reviews mapping."""
    try:
        genre_reviews_dict = pickle.load(open(cache_path, "rb"))
        print(f"Loaded genre reviews from cache: {cache_path}")
        return genre_reviews_dict
    except FileNotFoundError:
        pass

    genre_reviews_dict = {}
    for genre, url in genre_url_dict.items():
        print(f"Downloading reviews for genre: {genre}")
        genre_reviews_dict[genre] = load_reviews(url, head=head, sample_size=sample_size)

    pickle.dump(genre_reviews_dict, open(cache_path, "wb"))
    print(f"Saved genre reviews to cache: {cache_path}")
    return genre_reviews_dict


def split_data(
    genre_reviews_dict: dict,
    train_per_genre: int = 800,
    test_per_genre: int = 200,
    seed: int = 42,
):
    """Split reviews into train / test sets."""
    random.seed(seed)
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, reviews in genre_reviews_dict.items():
        reviews = random.sample(reviews, min(len(reviews), train_per_genre + test_per_genre))
        for review in reviews[:train_per_genre]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in reviews[train_per_genre:train_per_genre + test_per_genre]:
            test_texts.append(review)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels


def build_label_maps(labels: list) -> tuple:
    """Build label ↔ id mapping dicts."""
    unique_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class GoodreadsDataset(torch.utils.data.Dataset):
    """Torch Dataset wrapping HuggingFace tokenizer encodings + integer labels."""

    def __init__(self, encodings, labels: list):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)