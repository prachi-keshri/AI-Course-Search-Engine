
from preprocessing import load_documents
from indexing import build_index
from tfidf import build_tfidf
from similarity import calculate_similarities
from ranking import rank_documents
from baseline import keyword_frequency_search


def main():
    print("=" * 50)
    print("COURSE MATERIAL SEARCH ENGINE")
    print("TF-IDF + Cosine Similarity vs Keyword Frequency")
    print("=" * 50)

    # Step 1: Load and preprocess PDFs
    documents = load_documents(folder="dataset")

    if not documents:
        print("No PDF documents found in dataset folder.")
        return

    print(f"\nLoaded {len(documents)} documents.")

    # Step 2: Build vocabulary and inverted index
    index = build_index(documents)
    print(f"Vocabulary size: {len(index['vocabulary'])}")

    # Step 3: Calculate TF-IDF document vectors
    document_vectors, idf_values = build_tfidf(index)
    print("Indexing completed successfully!")

    # Search functions shared by interactive search
    # and experimental evaluation
    def tfidf_search(query):
        scores = calculate_similarities(
            query, document_vectors, idf_values
        )
        return rank_documents(scores, top_k=5)

    def baseline_search(query):
        return keyword_frequency_search(
            query, index, top_k=5
        )

    # Step 4: Interactive search
    while True:
        print("\n" + "-" * 50)
        query = input(
            "Enter search query (or 'exit'): "
        ).strip()

        if query.lower() == "exit":
            print("Thank you for using the search engine!")
            break

        if not query:
            print("Please enter a valid query.")
            continue

        # Step 5: Run both algorithms
        tfidf_results = tfidf_search(query)
        baseline_results = baseline_search(query)

        # Step 6: Display TF-IDF results
        print("\n--- TF-IDF + COSINE SIMILARITY ---")

        if not tfidf_results:
            print("No matching documents found.")
        else:
            for rank, (name, score) in enumerate(
                tfidf_results, start=1
            ):
                print(
                    f"{rank}. {name} | "
                    f"Similarity: {score:.6f}"
                )

        # Step 7: Display baseline results
        print("\n--- KEYWORD FREQUENCY BASELINE ---")

        if not baseline_results:
            print("No matching documents found.")
        else:
            for rank, (name, score) in enumerate(
                baseline_results, start=1
            ):
                print(
                    f"{rank}. {name} | "
                    f"Frequency: {score}"
                )

        print("-" * 50)


if __name__ == "__main__":
    main()
