from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login

from qdrant_client import QdrantClient
from qdrant_client.http import models

import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from pydantic import BaseModel, Field

from fastembed import SparseTextEmbedding
from uuid import uuid4
import tqdm
import random
import pandas as pd

from search_functions import semantic_search, hybrid_search
import pickle
from utils.user_query_augment import mqe, qae
import numpy as np
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


##Tf_idf and cross-encoder model loaded
bm25_model = SparseTextEmbedding("Qdrant/bm25")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
cross_encoder_model_name = "ms-marco-MiniLM-L-12-v2"

# embeddings = ["infly/inf-retriever-v1-1.5b", "BAAI/bge-small-en-v1.5", "Qwen/Qwen3-Embedding-0.6B"]
embeddings = [ "BAAI/bge-small-en-v1.5", "Qwen/Qwen3-Embedding-0.6B"]

search_techniques = ["hybrid", "semantic"]
cross_encoder_types = [False, True]
query_types = [None, "qae", "mqe"]


def precision(actual, predicted):

    total_predicted = len(predicted)
    correct = len(set(actual).intersection(set(predicted)))
    return correct/total_predicted


def recall(actual, predicted):

    total_actual = len(actual)
    correct = len(set(actual).intersection(set(predicted)))
    return correct/total_actual



def get_embedding(embedding_model):

        ##Embedding initialization
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )
        print("[Info] Embedding model is loaded!")
        return embeddings


def cross_encoder_func(query, documents, k=5):

    pairs = [[query, doc] for doc in documents]
    scores = cross_encoder.predict(pairs)
    top_indices = np.argsort(scores)[::-1][:k]
    top_k_documents = [documents[idx] for idx in top_indices]

    return top_k_documents


def compute_precision_recall(embedding_model, embeddings, query_type, search_type, cross_encoder_use, version):
    df = pd.read_csv('test_dataset.csv')
    final = []

    top_k = 10
    print(embedding_model,'|', query_type, '|', search_type, '|', cross_encoder_use,'|', version)
    for row, col in df.iterrows():

        query = col['question']
        if query_type == None:
             pass
        elif query_type == "qae":
             query = query+'\n'+qae(query)
        else:
             query = mqe(query)+'\n'+query
        
        if search_type == "hybrid":
            
            out = hybrid_search(client, f"insurance_hybrid_{embedding_model.split('/')[-1].replace('-', '_')}", query, embeddings, bm25_model, top_k=top_k)
        else:
            out = semantic_search(client, f"insurance_hybrid_{embedding_model.split('/')[-1].replace('-', '_')}", query, embeddings, top_k=top_k)

        chunks = [i['text'] for i in out]
        if cross_encoder_use:
            top_k = 5
            cross_out = cross_encoder_func(query, chunks, top_k)
            retrieved_context = str([i for i in cross_out])
        else:
            retrieved_context = str([i['text'] for i in out])
        final.append(retrieved_context)


    df['retrieved_context'] = final
    p = []
    r = []

    for row, col in df.iterrows():

        actual = eval(col['context'])
        predicted = eval(col['retrieved_context'])
        prec = precision(actual, predicted)
        rec = recall(actual, predicted)
        print(actual,'\n', predicted)
        print(prec, rec)
        p.append(prec)
        r.append(rec)
    

    df['precision'] = p
    df['recall'] = r

    df_type_log = df.groupby('question_type')[['precision', 'recall']].mean().reset_index()
    df_type_log['version'] = f'v{version}'

    df_log = pd.DataFrame()
    df_log['embedding'] = [embedding_model]
    df_log['user_q'] = ["no augmentation" if query_type == None else query_type]
    df_log['search_type'] = [search_type]
    df_log['top_k'] = [top_k]
    df_log['reranker'] = ["Yes" if cross_encoder_use else "No"]
    df_log['reranker_model'] = [cross_encoder_model_name if cross_encoder_use else "NA"]
    df_log['average_precision_pr'] = [df['precision'].mean()]
    df_log['average_recall_pr'] = [df['recall'].mean()]
    df_log['version'] = [f'v{version}']

    return df_type_log, df_log



version = 12
for embedding_model in tqdm.tqdm(embeddings, desc="Main iterator"):
    embeddings = get_embedding(embedding_model)
    for query_type in query_types:
            for search_type in search_techniques:
                for cross_enocder_use in cross_encoder_types:
                    df_type_log, df_log = compute_precision_recall(embedding_model, embeddings, query_type, search_type, cross_enocder_use, version)
                    print(f"Version[{version} is completed]")

                    wandb.init(
                        # set the wandb project where this run will be logged
                        project="RAG_EVALS",  name=f"experiment_v1",
                    )

                    wandb.log({"evaluation_table": df_log})


                    wandb.log({"question_type_log": df_type_log})
                    wandb.finish()

                    print(f"[Info] v{version} Metrics stored in wandb")
                    version += 1










