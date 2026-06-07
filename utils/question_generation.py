from langchain_huggingface import HuggingFaceEmbeddings

from qdrant_client import QdrantClient

import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate



from pydantic import BaseModel, Field
import tqdm
import random

from typing import Optional
from utils.user_query_augment import mqe, qae
import pandas as pd 

load_dotenv() 

# qdrant_url = os.getenv("QDRANT_URL")
# qdrant_api_key = os.getenv("QDRANT_API_KEY")

# embeddings = HuggingFaceEmbeddings(
#     model_name="infly/inf-retriever-v1-1.5b"
# )

# client = QdrantClient(
#     url=qdrant_url,
#     api_key=qdrant_api_key,
#     timeout = 800
# )


# # vectorstore = QdrantVectorStore(
# #             client=client,
# #             collection_name='insurance_hybrid_inf_retriever_v1_1.5b',
# #             embedding=embeddings
# #         )

# def vectorstore_to_dataframe(client: QdrantClient, collection_name: str, limit: Optional[int] = None, include_vectors: bool = False, vector_type: str = "dense") -> pd.DataFrame:
#     all_rows = []
#     offset = None
#     retrieved = 0

#     while True:
#         points, offset = client.scroll(
#             collection_name=collection_name,
#             limit=100,  # batch size
#             offset=offset,
#             with_vectors=include_vectors,
#             with_payload=True
#         )

#         if not points:
#             break

#         for point in points:
#             row = {"id": point.id}

#             # Payload (normalize keys)
#             if point.payload:
#                 for k, v in point.payload.items():
#                     row[str(k).lower()] = v

#             # Vectors
#             if include_vectors and point.vector:
#                 if isinstance(point.vector, dict):  # named vectors
#                     if vector_type in ["dense", "both"] and "dense" in point.vector:
#                         row["dense_vector"] = (
#                             point.vector["dense"].tolist()
#                             if hasattr(point.vector["dense"], "tolist")
#                             else point.vector["dense"]
#                         )
#                     if vector_type in ["sparse", "both"] and "bm25" in point.vector:
#                         sv = point.vector["bm25"]
#                         row["sparse_indices"] = sv.indices
#                         row["sparse_values"] = sv.values
#                 else:  # single unnamed vector
#                     row["vector"] = (
#                         point.vector.tolist()
#                         if hasattr(point.vector, "tolist")
#                         else point.vector
#                     )

#             all_rows.append(row)

#         retrieved += len(points)
#         if limit and retrieved >= limit:
#             break
#         if offset is None:
#             break

#     df = pd.DataFrame(all_rows)
#     print(f"✅ Retrieved {len(df)} records from collection '{collection_name}'")
#     return df

# df = vectorstore_to_dataframe(client, "insurance_hybrid_inf_retriever_v1_1.5b", limit=150, include_vectors=True, vector_type="both")
# df = df[['text', 'context']] 

# df.to_csv("text_context.csv", index = False)
# print(f"Context is stored inside a csv file")

df = pd.read_csv("text_context.csv")
def get_random_n_rows(df, column_name, n):
    return df[column_name].sample(n=n).tolist() 

llama ='llama-3.3-70b-versatile'
model = ChatGroq(model_name = llama, temperature = 0)

def random_question_type():
    x = {
        'simple question': '''
Simple questions generated from an excerpt of the knowledge base
Example: What is the capital of France?
''',

        'distracting question': '''
Questions made to confuse the retrieval part of the RAG with a distracting element from the knowledge base but irrelevant to the question
Example: Italy is beautiful but what is the capital of France?
''',
        'double question': '''
Questions with two distinct parts to evaluate the capabilities of the query rewriter of the RAG
Example: What is the capital and the population of France?
''',
        'conversational question': '''
Questions made as part of a conversation, first message describes the context of the question that is asked in the last message, also tests the rewriter
Example: (two separate messages)
I would like to know some information about France.
What is its capital city?
'''
    }

    random_key = random.choice(list(x.keys()))

    return (random_key, x[random_key].strip())

class QuestionAnswer(BaseModel):
    question: str = Field(description="The generated question based on the context")
    answer: str = Field(description="The generated answer for the question")
    

parser = PydanticOutputParser(pydantic_object=QuestionAnswer)

def generate_questions(context, question_nudge, question='Please generate question answer pair'):
    # Create the parser
    
    # Get format instructions
    format_instructions = parser.get_format_instructions()
    
    template = ChatPromptTemplate.from_messages([
        ("system", """
         You are Health insurance question generator
         - You are a helpful Question Answer generator, where you will generate ONLY one question and answer pair based on context. Context here is based on health insurance.
         - You generate complete questions from CONTEXT ONLY BUT PATTERN of asking quesions SHOULD be AS PER question nudge. STRICTLY FOLLOW QUESTION NUDGE PATTERN of asking questions
         - Context will be in form of list or list or lists. If question nudge type is double question, then please generate question from BOTH the lists given in context wherever possible.
         - please generate convesational questions from question nudge as per described pattern in example. This is very important.
        ** While generating questions as per mentioned rules, make sure to geterate questions like how a person that has health insurance will ask questions to health insurance help deskwhen it is needed. THIS IS MANDATORY

         Your response must be in the following format WITHOUT ANY additional explanation or text:
         {format_instructions}

         Context:
         {context}

         Question nudge:
         {question_nudge}
"""),
        ("human", "{question}")
    ]) 

    chain = (
        RunnableMap({
            "context": lambda x: context,
            "question_nudge": lambda x: question_nudge,
            "question": lambda x: x['question'],
            "format_instructions": lambda x: format_instructions
        }) 
        
        | template 
        | model 
        | parser

    )

    response = chain.invoke({'question': question})
    return response

testing_final = []
for i in tqdm.tqdm(range(50)):
    q_type, exple = random_question_type()

    if q_type == 'double question':
        random_number=2
    else:
        random_number=1

    #random_number = random.choices([1, 2], weights=[0.6, 0.4])[0]
    x = get_random_n_rows(df, 'text', random_number)  #[[q1, q2]]
    final_x = []
    for sent in x:
        if len(sent.strip().split())>20:
            final_x.append(sent.strip())

    if len(final_x)<1:
        continue

    if len(str(x).split())<30:
        continue


    final = generate_questions(str(final_x), q_type+':'+exple)

    testing_final.append([q_type, final.question, final_x, final.answer])

df_test = pd.DataFrame(testing_final, columns=['question_type', 'question', 'context', 'answer'])
df_test.to_csv('test_dataset.csv', index = False)