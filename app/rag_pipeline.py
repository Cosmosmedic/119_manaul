from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from app.loader import load_manual_pages


class ManualRAG:
    def __init__(self):
        print("📌 Loading Embedding Model...")
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    def build_index(self):
        print("📌 Loading manual pages...")
        self.pages = load_manual_pages()

        texts = [p["text"] for p in self.pages]

        print("📌 Generating embeddings...")
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)

        self.index = index
        self.embeddings = embeddings

        print(f"📌 Index built with {len(texts)} pages")

    def search(self, query, top_k=5):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        scores, ids = self.index.search(q_emb, top_k)

        results = []

        for score, idx in zip(scores[0], ids[0]):
            results.append({
                "page": self.pages[idx]["page"],
                "text": self.pages[idx]["text"],
                "score": float(score)
            })

        return results

