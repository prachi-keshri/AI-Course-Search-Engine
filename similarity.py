
import math

from preprocessing import preprocess_text, load_documents
from indexing import build_index
from tfidf import build_tfidf, calculate_tf
from collections import Counter


def build_query_vector(query, idf_values):
    tokens = preprocess_text(query)
    counts = Counter(tokens)

    query_vector = {}

    for term, count in counts.items():
        if term in idf_values:
            tf = calculate_tf(count, len(tokens))
            query_vector[term] = tf * idf_values[term]

    return query_vector


def cosine_similarity(query_vector, document_vector):
    dot_product = 0.0

    for term, query_weight in query_vector.items():
        document_weight = document_vector.get(term, 0.0)
        dot_product += query_weight * document_weight

    query_magnitude = math.sqrt(
        sum(weight ** 2 for weight in query_vector.values())
    )

    document_magnitude = math.sqrt(
        sum(weight ** 2 for weight in document_vector.values())
    )

    if query_magnitude == 0 or document_magnitude == 0:
        return 0.0

    return dot_product / (query_magnitude * document_magnitude)


def calculate_similarities(query, document_vectors, idf_values):
    query_vector = build_query_vector(query, idf_values)

    scores = {}

    for doc_name, document_vector in document_vectors.items():
        scores[doc_name] = cosine_similarity(
            query_vector, document_vector
        )

    return scores


if __name__ == "__main__":
    documents = load_documents(folder="dataset")
    index = build_index(documents)
    document_vectors, idf_values = build_tfidf(index)

    query = input("Enter your search query: ")

    scores = calculate_similarities(
        query, document_vectors, idf_values
    )

    print("\nSimilarity scores:")

    for doc_name, score in sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"{doc_name}: {score:.6f}")
