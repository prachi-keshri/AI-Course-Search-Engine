
import re
from pathlib import Path
import pymupdf

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "in", "on", "at", "to", "for", "of", "and",
    "or", "with", "by", "as", "from", "that",
    "this", "it", "be", "has", "have", "had"
}

def preprocess_text(text):
    text = text.lower()
    words = re.findall(r"[a-z]+", text)
    return [word for word in words if word not in STOP_WORDS]


def load_documents(folder="dataset"):
    documents = {}

    for pdf_path in sorted(Path(folder).glob("*.pdf")):
        with pymupdf.open(pdf_path) as pdf:
            text = ""
            for page in pdf:
                text += page.get_text()

        tokens = preprocess_text(text)
        documents[pdf_path.name] = tokens

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print("Total documents loaded:", len(documents))

    for name, tokens in documents.items():
        print(f"{name}: {len(tokens)} tokens")
        print("Sample tokens:", tokens[:10])
        print()
