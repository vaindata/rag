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
Based on Hughes Hallucination Evaluation Model (HHEM) Leaderboard and Open source criteria These models have been filtered out.

| Rank | Model                                    | Hallucination Score (%) | Context Window |
|------|------------------------------------------|-------------------------|----------------|
| 1    | microsoft/Phi-4                          | 3.7%                    | 16K tokens     |
| 2    | meta-llama/Llama-3.3-70B-Instruct-Turbo  | 4.1%                    | 128K tokens    |
| 3    | snowflake/snowflake-arctic-instruct      | 4.3%                    | 4K tokens      |
| 4    | google/gemma-3-12b-it                    | 4.4%                    | 128K tokens    |
| 5    | mistralai/mistral-large-2411             | 4.5%                    | 128K tokens    |

Although ideally hallucination score must be less than 3% ideally, but due to computational constraint and ease of use we have taken two models from Groq(llama-3.1-8b-instant and llama-3.3-70b-versatile) and OpenAI(gpt-4o, gpt-4o-mini) also our context window desired length is 128K. These four language models were evaluated on a set of 10 question-answer pairs.
Statistics of these models are mentioned below.

| Rank | LLM                     | Context Window | Hughes Hallucination Score (%) | Open Source or Commercial |
|------|-------------------------|----------------|-------------------------------|---------------------------|
| 1    | llama-3.3-70b-versatile | 128K           | 4.1                           | Open Source               |
| 2    | gpt-4o                  | 128K           | 9.6                           | Commercial                |
| 3    | gpt-4o-mini             | 128K           | Not publicly listed           | Commercial                |
| 4    | llama-3.1-8b-instant    | 128K           | Not publicly listed           | Open Source               |

#### Evaluation Metrics

- Faithfulness
- Noise Sensitivity

## Experiment Result

| Rank | llm                    | average_faithfulness | average_noise_sensitivity |
|------|------------------------|----------------------|---------------------------|
| 1    | gpt-4o-mini            | 0.9282               | 0.2860                    |
| 2    | gpt-4o                 | 0.8649               | 0.2564                    |
| 3    | llama-3.3-70b-versatile | 0.7781               | 0.3010                    |
| 4    | llama-3.1-8b-instant   | 0.6956               | 0.3630                    |

#### Key Findings

| Model | Observation |
|---------|-------------|
| gpt-4o-mini | Highest average Faithfulness and Noise Sensitivity scores |
| Llama 3.3 70B | Best-performing open-source model |

Although GPT-4o-mini achieved the strongest performance, deployment constraints led to the selection of Llama 3.3 70B as the final generation model.

#### Experiment Dashboard

GEN_EVALS W&B Project:

https://wandb.ai/debojyoti-temporary1-indian-statistical-institute/GEN_EVALS?nw=nwuserdebojyotitemporary1