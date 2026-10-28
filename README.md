
# AI Course Material Search Engine

A Python-based search engine that retrieves relevant course material PDFs using information retrieval techniques.

## Features
- PDF text extraction and preprocessing
- Vocabulary and inverted index construction
- TF-IDF weighting and cosine similarity
- Keyword-frequency baseline for comparison
- Top-5 document retrieval
- Experimental evaluation using Precision@5, MRR, and query execution time

## Technologies
- Python
- PyMuPDF

## Dataset
The project uses 34 course material PDFs across 9 subjects.

The PDFs are excluded from this repository. Users can add their own PDFs to a folder named `dataset`.

## How to Run

1. Install dependencies:
   `pip install -r requirements.txt`

2. Create a `dataset` folder and add PDF files.

3. Run the interactive search engine:
   `python main.py`

4. Run the evaluation:
   `python run_experiments.py`

   Evaluation metrics require relevance labels in `results/relevance_labels.json`.

## Algorithms
1. Keyword-frequency ranking
2. TF-IDF with cosine similarity

## Evaluation
Both algorithms are compared using Precision@5, Mean Reciprocal Rank (MRR), and average query execution time.

The current relevance judgments are provisional and based on subject-level matching rather than independently verified document relevance.
