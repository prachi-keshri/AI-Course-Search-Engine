import math
from indexing import build_index
from preprocessing import load_documents


def calculate_tf(term_count, document_length):
    if document_length == 0:
        return 0.0

    return term_count / document_length


def calculate_idf(total_documents, document_frequency):
    if total_documents == 0 or document_frequency == 0:
        return 0.0

    return math.log(total_documents / document_frequency)


def build_tfidf(index):
    total_documents = index["total_documents"]
    document_frequency = index["document_frequency"]

    idf_values = {}

    for term in index["vocabulary"]:
        idf_values[term] = calculate_idf(
            total_documents,
            document_frequency[term]
        )

    document_vectors = {}

    for doc_name, counts in index["term_counts"].items():
        document_length = index["document_lengths"][doc_name]
        vector = {}

        for term, count in counts.items():
            tf = calculate_tf(count, document_length)
            idf = idf_values[term]

            vector[term] = tf * idf

        document_vectors[doc_name] = vector

    return document_vectors, idf_values


if __name__ == "__main__":
    documents = load_documents(folder="dataset")
    index = build_index(documents)

    document_vectors, idf_values = build_tfidf(index)

    print("TF-IDF vectors created:", len(document_vectors))

    for doc_name, vector in document_vectors.items():
        print("\nDocument:", doc_name)

        top_terms = sorted(
            vector.items(),
            key=lambda item: item[1],
            reverse=True
        )[:5]

        for term, weight in top_terms:
            print(f"{term}: {weight:.6f}")

        break
