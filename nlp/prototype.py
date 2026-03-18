import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TextPrototype:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.model = LogisticRegression(max_iter=500)
        self.label_to_idx = {}
        self.idx_to_label = {}

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def fit(self, texts, labels):
        clean_texts = [self.clean_text(t) for t in texts]
        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}

        y = [self.label_to_idx[label] for label in labels]
        x = self.vectorizer.fit_transform(clean_texts)
        self.model.fit(x, y)

    def predict(self, text: str):
        clean = self.clean_text(text)
        x = self.vectorizer.transform([clean])
        pred_idx = int(self.model.predict(x)[0])
        return self.idx_to_label[pred_idx]


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
        "status": "NLP scaffold working: clean -> vectorize -> classify",
    }

    out_path = Path("metrics/nlp_prototype_output.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    demo()
