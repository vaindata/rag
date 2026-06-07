# Getting Started

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the RAG Pipeline

To execute the RAG pipeline:

```bash
python rag_pipeline.py
```

## Project Structure

```text
.
├── rag_pipeline.py        # Main pipeline script
├── requirements.txt       # Project dependencies
├── data/                  # Source documents (if applicable)
├── bias report/   ## csv files for three embedding models
└── README.md
└── CONCLUSION.md
```

## Notes

- SEE CONCLUSION.md for detailed understanding.
- The pipeline is designed to run on CPU-only environments.
- Experiments were conducted on a system with 8 GB RAM and no GPU.
- The selected retrieval configuration uses:
  - Embedding Model: `BAAI/bge-small-en-v1.5`
  - Search Strategy: `Hybrid Search`
  - Reranker: `ms-marco-MiniLM-L-12-v2`
- The selected generation model is `Llama 3.3 70B`.