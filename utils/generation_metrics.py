from ragas.llms import LangchainLLMWrapper
# from groq import AsyncGroq
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.dataset_schema import SingleTurnSample 
from ragas.metrics.collections import Faithfulness, NoiseSensitivity

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from search_functions import hybrid_search
from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder
import os 
import pandas as pd 
import numpy as np
from qdrant_client import QdrantClient
from dotenv import load_dotenv
load_dotenv()
import wandb

wandb.login(key=os.getenv("WANDB_API_KEY"))

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
client = QdrantClient( #Intializing client
    url=qdrant_url,
    api_key=qdrant_api_key,
    timeout = 800
)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
cross_encoder_model_name = "ms-marco-MiniLM-L-12-v2"

def get_embedding(embedding_model):

    ##Embedding initialization
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model
    )
    print("[Info] Embedding model is loaded!")
    return embeddings

embedding_model = "BAAI/bge-small-en-v1.5"
embeddings = get_embedding(embedding_model)
bm25_model = SparseTextEmbedding("Qdrant/bm25")

df = pd.read_csv('test_dataset.csv').iloc[:10] 

# llm = 'gpt-4o'
# llm = "llama-3.1-8b-instant"
llm = "llama-3.3-70b-versatile"
model = ChatGroq(temperature=0, model_name=llm)
# model = ChatOpenAI(model =llm, temperature = 0)
openai_client = AsyncOpenAI()
# openai_client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
evaluator_llm = llm_factory('gpt-4o-mini', client=openai_client)

# groq_client = AsyncGroq(
#     api_key=os.getenv("GROQ_API_KEY"))
# evaluator_llm = llm_factory(model=llama, client=groq_client)


prompt_str = '''
You are a helpful AI assistant that assists humans on health insurance ONLY based on the knowledge provided below.

Knowledge:
'''

template = ChatPromptTemplate.from_messages([
        ("system", f"{prompt_str}\n{'{context}'}"),
        ("human", "{question}")
    ])

def predict(question, context):

    chain = (
        RunnableMap({
            "context": lambda x: context,  # Use retrieved context
            "question": lambda x: x["question"]
        })
        | template 
        | model 
        | StrOutputParser()
    )

    llm_response = chain.invoke({"question": question})

    return llm_response

def cross_encoder_func(query, documents, k=5):

    pairs = [[query, doc] for doc in documents]
    scores = cross_encoder.predict(pairs)
    top_indices = np.argsort(scores)[::-1][:k]
    top_k_documents = [documents[idx] for idx in top_indices]

    return top_k_documents

final = []
retrieved_contexts = []
import tqdm
for row, col in tqdm.tqdm(df.iterrows()):

    query = col['question']

    out = hybrid_search(client, f"insurance_hybrid_{embedding_model.split('/')[-1].replace('-', '_')}", query, embeddings, bm25_model, top_k=10)
    chunks = [i['text'] for i in out]
    cross_out = cross_encoder_func(query, chunks, k = 5)
    retrieved_context = str([i for i in cross_out])
    retrieved_contexts.append(retrieved_context)
    response = predict(query, retrieved_context)

    final.append(response)

df["retrieved_context"] = retrieved_contexts
df['predicted_response'] = final
df["llm"] = llm

async def faithfulness(query, context, response, llm):
    scorer = Faithfulness(llm=llm)
    
    result = await scorer.ascore(
        user_input=query,
        response=response,
        retrieved_contexts=context
    )
    
    return result.value if hasattr(result, "value") else result


async def noise_sensitivity(query, context, response, reference, llm):
    scorer = NoiseSensitivity(llm=llm)
    
    result = await scorer.ascore(
        user_input=query,
        response=response,
        reference=reference,
        retrieved_contexts=context
    )
    
    return result.value if hasattr(result, "value") else result
    

import asyncio

final_eval = []
for row, col in tqdm.tqdm(df.iterrows()):
    query = col['question']
    context = eval(col['retrieved_context'])
    response = col['predicted_response']
    
    f = asyncio.run(faithfulness(query, context, response, evaluator_llm))

    final_eval.append(f)
df['faithfulness_score'] = final_eval

final_eval = []
for row, col in tqdm.tqdm(df.iterrows()):
    query = col['question']
    context = eval(col['retrieved_context'])
    response = col['predicted_response']
    ref = col['answer']
    
    f = asyncio.run(noise_sensitivity(query, context, response, ref, evaluator_llm))

    final_eval.append(f)
df['noise_sensitivity_score'] = final_eval
df_table = df[["faithfulness_score", "noise_sensitivity_score", "llm"]]

df_log = pd.DataFrame()
df_log['average_faithfulness'] = [df['faithfulness_score'].mean()]
df_log['average_noise_sensitivity'] = [df['noise_sensitivity_score'].mean()]
df_log['llm'] = [llm]
wandb.init(
    # set the wandb project where this run will be logged
    project="GEN_EVALS",  name=f"experiment_v1",
)

wandb.log({"evaluation_table": df_table})
wandb.log({"evaluation_log": df_log})
wandb.finish()
# print(df_log)

print("Evaluation completed and logged to Weights & Biases!")