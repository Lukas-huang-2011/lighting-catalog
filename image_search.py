"""
Image similarity search using perceptual hashing.
Fast, free, no heavy ML models required.
"""

import imagehash
from PIL import Image


def compute_hash(image: Image.Image) -> str:
    """Compute a perceptual hash string for an image."""
    return str(imagehash.phash(image, hash_size=16))


def hash_distance(hash1: str, hash2: str) -> int:
    """Compute the Hamming distance between two hash strings."""
    try:
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return h1 - h2
    except Exception:
        return 9999


def find_similar(query_image: Image.Image, stored: list, threshold: int = 20) -> list:
    """
    Find stored images similar to the query image.

    Args:
        query_image: PIL Image to search for
        stored: list of dicts, each must have 'image_hash' key
        threshold: max hash distance (lower = stricter). 20 allows slight variations.

    Returns:
        List of matching dicts sorted by similarity (best first), with 'similarity_score' added.
    """
    query_hash = compute_hash(query_image)
    results = []

    for item in stored:
        if not item.get("image_hash"):
            continue
        dist = hash_distance(query_hash, item["image_hash"])
        if dist <= threshold:
            results.append({**item, "similarity_score": max(0, 100 - dist)})

    return sorted(results, key=lambda x: x["similarity_score"], reverse=True)
