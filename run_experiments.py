
import csv
import json
from pathlib import Path
from collections import Counter
from time import perf_counter

from preprocessing import load_documents
from indexing import build_index
from tfidf import build_tfidf
from similarity import calculate_similarities
from ranking import rank_documents
from baseline import keyword_frequency_search
from evaluation import evaluate_method


DATASET = "dataset"
OUTPUT = Path("results")
LABEL_FILE = OUTPUT / "relevance_labels.json"
TOP_K = 5

# Queries designed to cover the final nine subjects.
# Verify their relevance against the actual PDF contents.
QUERIES = [
    "artificial intelligence search algorithms",
    "knowledge representation reasoning",
    "machine learning classification",
    "supervised learning regression",
    "database normalization",
    "sql queries database",
    "data structures trees",
    "sorting algorithms complexity",
    "process scheduling operating systems",
    "deadlock memory management",
    "software engineering testing",
    "software development life cycle",
    "microprocessor instruction set",
    "embedded systems architecture",
    "aws cloud computing"
]


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_labels(documents):
    """
    Creates a labeling template if one does not exist.
    Edit the JSON to list genuinely relevant PDFs for each query.
    """
    if not LABEL_FILE.exists():
        labels = {
            query: [] for query in QUERIES
        }
        LABEL_FILE.write_text(
            json.dumps(labels, indent=2),
            encoding="utf-8"
        )
        print(f"\nCreated {LABEL_FILE}")
        print("Add relevant PDF filenames before final evaluation.")
        return None

    labels = json.loads(LABEL_FILE.read_text(encoding="utf-8"))

    if set(labels) != set(QUERIES):
        raise ValueError(
            "Query list and relevance_labels.json do not match."
        )

    valid_docs = set(documents)
    for query, filenames in labels.items():
        if not filenames:
            print(f"Missing relevance labels: {query}")
            return None

        invalid = set(filenames) - valid_docs
        if invalid:
            raise ValueError(
                f"Unknown filenames for '{query}': {invalid}"
            )

    return {
        query: set(filenames)
        for query, filenames in labels.items()
    }


def main():
    OUTPUT.mkdir(exist_ok=True)

    print("Loading final dataset...")
    documents = load_documents(folder=DATASET)

    if not documents:
        print("No documents found.")
        return

    print(f"Loaded {len(documents)} PDFs.")

    index = build_index(documents)
    document_vectors, idf_values = build_tfidf(index)

    def baseline_search(query):
        return keyword_frequency_search(query, index, top_k=TOP_K)

    def tfidf_search(query):
        scores = calculate_similarities(
            query, document_vectors, idf_values
        )
        return rank_documents(scores, top_k=TOP_K)

    methods = {
        "Keyword Frequency": baseline_search,
        "TF-IDF + Cosine": tfidf_search
    }

    # Save all query rankings for relevance labeling.
    ranking_rows = []
    for query in QUERIES:
        for method, search in methods.items():
            for rank, (doc, score) in enumerate(search(query), 1):
                ranking_rows.append({
                    "Algorithm": method,
                    "Query": query,
                    "Rank": rank,
                    "Document": doc,
                    "Score": score
                })

    write_csv(
        OUTPUT / "rankings.csv",
        ["Algorithm", "Query", "Rank", "Document", "Score"],
        ranking_rows
    )

    labels = prepare_labels(documents)

    if labels is None:
        print("\nRankings exported successfully.")
        print("Review rankings.csv and complete relevance_labels.json.")
        print("Then run this script again for final metrics.")
        return

    summary_rows = []
    query_rows = []

    for method, search in methods.items():
        metrics = evaluate_method(search, labels, k=TOP_K)
        summary_rows.append({
            "Algorithm": method,
            **metrics
        })

        for query, relevant in labels.items():
            start = perf_counter()
            results = search(query)
            elapsed = (perf_counter() - start) * 1000

            hits = sum(
                doc in relevant for doc, _ in results[:TOP_K]
            )

            rr = next(
                (
                    1 / rank
                    for rank, (doc, _) in enumerate(results, 1)
                    if doc in relevant
                ),
                0.0
            )

            query_rows.append({
                "Algorithm": method,
                "Query": query,
                "Precision@5": hits / TOP_K,
                "Reciprocal Rank": rr,
                "Time (ms)": elapsed
            })

    write_csv(
        OUTPUT / "comparison.csv",
        ["Algorithm", "Precision@5", "MRR", "Average Time (ms)"],
        summary_rows
    )

    write_csv(
        OUTPUT / "query_results.csv",
        [
            "Algorithm", "Query", "Precision@5",
            "Reciprocal Rank", "Time (ms)"
        ],
        query_rows
    )

    print("\nFINAL EXPERIMENTAL RESULTS")
    print("-" * 60)
    for row in summary_rows:
        print(row)

    print("\nAll results saved to:", OUTPUT)


if __name__ == "__main__":
    main()
