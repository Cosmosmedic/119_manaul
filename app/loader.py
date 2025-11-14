import json
import os
from PIL import Image


def load_manual_pages(path="data/manual_pages.jsonl"):
    pages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))
    return pages


def load_sections(path="data/manual_sections_hybrid.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_algorithms(path="data/algorithm_pages.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_algorithm_image(page):
    img_path = f"data/images/alg_{page}.png"
    if os.path.exists(img_path):
        return Image.open(img_path)
    return None

