
def keyword_frequency_search(query, index, top_k=5):
    """
    Rank documents by total frequency of query terms.
    Uses the existing inverted index.
    """
    from preprocessing import preprocess_text

    query_tokens = preprocess_text(query)
    scores = {}

    for term in query_tokens:
        postings = index["inverted_index"].get(term, {})

        for doc_name, frequency in postings.items():
            scores[doc_name] = scores.get(doc_name, 0) + frequency

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0])
    )[:top_k]
