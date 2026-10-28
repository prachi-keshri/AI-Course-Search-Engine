
def rank_documents(scores, top_k=5):
    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0])
    )

    # Exclude documents with zero similarity
    ranked = [
        (name, score)
        for name, score in ranked
        if score > 0
    ]

    return ranked[:top_k]


if __name__ == "__main__":
    sample_scores = {
        "OS_part1.pdf": 0.15,
        "DBMS_part1.pdf": 0.03,
        "MPMC_part1.pdf": 0.42,
        "AI_part1.pdf": 0.00
    }

    results = rank_documents(sample_scores, top_k=3)

    for rank, (name, score) in enumerate(results, start=1):
        print(f"{rank}. {name} | Score: {score:.6f}")
