from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from utils.search_functions import hybrid_search
from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder
import os 
import pandas as pd 
import numpy as np
from qdrant_client import QdrantClient
from dotenv import load_dotenv
load_dotenv()

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

llm = "llama-3.3-70b-versatile"
model = ChatGroq(temperature=0, model_name=llm)


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

if __name__ == "__main__":
    while True:
        query = input("Enter your health insurance question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        retrieved_docs = hybrid_search(client, f"insurance_hybrid_{embedding_model.split('/')[-1].replace('-', '_')}", query, embeddings, bm25_model, top_k=10)
        chunks = [i['text'] for i in retrieved_docs]
        for chunk in chunks:
            print(f"Retrieved chunk: {chunk}\n")
            print("-" * 50)
        top_k_docs = cross_encoder_func(query, chunks, k=5)
        context = "\n".join(top_k_docs)
        
        answer = predict(query, context)
        print(f"Answer: {answer}\n")

