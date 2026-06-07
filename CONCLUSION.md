## Experiment Results

### Retrieval Evaluation (RAG_EVALS)

The retrieval pipeline was evaluated across multiple combinations of:

- Embedding models
- Query augmentation strategies
- Search methods
- Reranking configurations

#### Embedding Models Evaluated

| Model |
|---------|
| BAAI/bge-small-en-v1.5 |
| Qwen/Qwen3-Embedding-0.6B |
| infly/inf-retriever-v1-1.5b |

#### Best Retrieval Configuration

| Parameter | Selected Value |
|------------|----------------|
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Query Augmentation | None |
| Search Method | Hybrid Search |
| Top_k Docs | 5 |
| Reranker | ms-marco-MiniLM-L-12-v2 |

#### Best Retrieval Metrics

| Metric | Score |
|---------|--------|
| Precision | 0.2326 |
| Recall | 0.9651 |

#### Experiment Dashboard

RAG_EVALS W&B Project:

https://wandb.ai/debojyoti-temporary1-indian-statistical-institute/RAG_EVALS?nw=nwuserdebojyotitemporary1

---

### Generation Evaluation (GEN_EVALS)

Four language models were evaluated on a set of 10 question-answer pairs.

#### Evaluation Metrics

- Faithfulness
- Noise Sensitivity

#### Key Findings

| Model | Observation |
|---------|-------------|
| gpt-4o-mini | Highest average Faithfulness and Noise Sensitivity scores |
| Llama 3.3 70B | Best-performing open-source model |

Although GPT-4o-mini achieved the strongest performance, deployment constraints led to the selection of Llama 3.3 70B as the final generation model.

#### Experiment Dashboard

GEN_EVALS W&B Project:

https://wandb.ai/debojyoti-temporary1-indian-statistical-institute/GEN_EVALS?nw=nwuserdebojyotitemporary1