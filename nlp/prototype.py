import json
import re
from collections import Counter
from pathlib import Path


class TextPrototype:
    def __init__(self):
        self.label_token_counts = {}
        self.label_doc_counts = {}
        self.global_token_counts = Counter()
        self.total_docs = 0

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def tokenize(text: str):
        return [tok for tok in text.split(" ") if tok]

    def fit(self, texts, labels):
        self.label_token_counts.clear()
        self.label_doc_counts.clear()
        self.global_token_counts.clear()
        self.total_docs = len(texts)

        for text, label in zip(texts, labels):
            clean = self.clean_text(text)
            tokens = self.tokenize(clean)
            if label not in self.label_token_counts:
                self.label_token_counts[label] = Counter()
                self.label_doc_counts[label] = 0

            self.label_token_counts[label].update(tokens)
            self.label_doc_counts[label] += 1
            self.global_token_counts.update(tokens)

    def score_label(self, tokens, label):
        token_counts = self.label_token_counts[label]
        total_label_tokens = sum(token_counts.values())
        vocab_size = max(1, len(self.global_token_counts))
        score = 0.0

        # Naive Bayes style log-prob score with Laplace smoothing.
        for token in tokens:
            prob = (token_counts[token] + 1.0) / (total_label_tokens + vocab_size)
            score += prob

        prior = self.label_doc_counts[label] / max(1, self.total_docs)
        score += prior
        return score

    def predict(self, text: str):
        clean = self.clean_text(text)
        tokens = self.tokenize(clean)
        if not self.label_token_counts:
            raise RuntimeError("Model must be fitted before prediction")

        best_label = None
        best_score = float("-inf")
        for label in sorted(self.label_token_counts.keys()):
            score = self.score_label(tokens, label)
            if score > best_score:
                best_score = score
                best_label = label
        return best_label


def demo():
    texts = [
        "surface has multiple linear scratches",
        "dark inclusion visible in steel",
        "small pits clustered in area",
        "rolled scale appears along edge",
        "patch-like texture irregularity",
        "fine crazing pattern observed",
    ]
    labels = [
        "scratches",
        "inclusion",
        "pitted_surface",
        "rolled_in_scale",
        "patches",
        "crazing",
    ]

    pipeline = TextPrototype()
    pipeline.fit(texts, labels)

    sample = "sample shows long scratch marks on metal sheet"
    pred = pipeline.predict(sample)

    result = {
        "sample_input": sample,
        "predicted_label": pred,
        "status": "NLP scaffold working: clean -> tokenize -> classify",
    }

    out_path = Path("metrics/nlp_prototype_output.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    demo()
