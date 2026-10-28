
from time import perf_counter


def precision_at_k(results, relevant_docs, k=5):
    """Fraction of top-k positions occupied by relevant documents."""
    retrieved = [doc for doc, _ in results[:k]]
    hits = sum(doc in relevant_docs for doc in retrieved)
    return hits / k


def reciprocal_rank(results, relevant_docs):
    """Reciprocal rank of the first relevant retrieved document."""
    for rank, (doc, _) in enumerate(results, start=1):
        if doc in relevant_docs:
            return 1 / rank
    return 0.0


def evaluate_method(search_function, test_queries, k=5):
    """
    search_function(query) -> [(filename, score), ...]
    test_queries = {
        "query text": {"relevant_file1.pdf", "relevant_file2.pdf"}
    }
    """
    precisions = []
    reciprocal_ranks = []
    execution_times = []

    for query, relevant_docs in test_queries.items():
        start = perf_counter()
        results = search_function(query)
        elapsed = (perf_counter() - start) * 1000

        precisions.append(
            precision_at_k(results, relevant_docs, k)
        )
        reciprocal_ranks.append(
            reciprocal_rank(results, relevant_docs)
        )
        execution_times.append(elapsed)

    n = len(test_queries)
    if n == 0:
        raise ValueError("At least one test query is required.")

    return {
        "Precision@5": sum(precisions) / n if k == 5 else sum(precisions) / n,
        "MRR": sum(reciprocal_ranks) / n,
        "Average Time (ms)": sum(execution_times) / n
    }
