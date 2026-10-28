
from collections import Counter, defaultdict
from preprocessing import load_documents


def build_index(documents):
    vocabulary = set()
    inverted_index = defaultdict(dict)
    term_counts = {}
    document_frequency = Counter()
    document_lengths = {}

    for doc_name, tokens in documents.items():
        counts = Counter(tokens)

        term_counts[doc_name] = dict(counts)
        document_lengths[doc_name] = len(tokens)
        vocabulary.update(counts.keys())

        for term, count in counts.items():
            inverted_index[term][doc_name] = count
            document_frequency[term] += 1

    return {
        "vocabulary": sorted(vocabulary),
        "inverted_index": dict(inverted_index),
        "term_counts": term_counts,
        "document_frequency": dict(document_frequency),
        "document_lengths": document_lengths,
        "total_documents": len(documents)
    }


if __name__ == "__main__":
    documents = load_documents(folder="dataset")
    index = build_index(documents)

    print("Documents indexed:", index["total_documents"])
    print("Vocabulary size:", len(index["vocabulary"]))

    print("\nFirst 15 vocabulary terms:")
    print(index["vocabulary"][:15])

    print("\nSample inverted index entries:")
    for term in ["process", "database", "network"]:
        print(f"{term}: {index['inverted_index'].get(term, {})}")
        print(f"Document frequency: {index['document_frequency'].get(term, 0)}")
