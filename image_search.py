"""Image similarity search using perceptual hashing."""

import imagehash
from PIL import Image


def compute_hash(image: Image.Image) -> str:
    return str(imagehash.phash(image, hash_size=16))


def hash_distance(hash1: str, hash2: str) -> int:
    try:
        return imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2)
    except Exception:
        return 9999


def find_similar(query_image: Image.Image, stored: list, threshold: int = 20) -> list:
    query_hash = compute_hash(query_image)
    results = []
    for item in stored:
        if not item.get("image_hash"):
            continue
        dist = hash_distance(query_hash, item["image_hash"])
        if dist <= threshold:
            results.append({**item, "similarity_score": max(0, 100 - dist)})
    return sorted(results, key=lambda x: x["similarity_score"], reverse=True)
