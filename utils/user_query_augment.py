import os
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate   # type: ignore
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
load_dotenv()


llama ='llama-3.3-70b-versatile'

# model = ChatGroq(temperature=0, model_name=llama)
model = ChatOpenAI(model ="gpt-4o-mini", temperature = 0)

def mqe(query):

    prompt = ChatPromptTemplate.from_template('''
                    You are a helpful expert health insurance research assistant. 
                    Given a user question please generate 3 questions related to the user question so that if i combine all these question there is high chance for me to retrive right results from vector database.
                    While generating questions make sure you ask questions like how people who bought insurance will ask the questions to zn insurance agent.
                                              
                    I just need 3 questions in output, i DONT want to see any explanation or any additional lines or any other verbose in the output.
                    I also DONT need something like "Here are three related questions:" in the output.

                    User_question:
                    {user_question}     
                                              """
                                                
                                                ''')

    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    response = chain.invoke({'user_question':query})

    return response



def qae(query):

    prompt = ChatPromptTemplate.from_template('''
                    You are a helpful expert insurance research assistant. 
                    Given a user question please generate a sample answer related to the user question.
                    Its ok if you dont know the exact true answer, i'm looking to get a sense of how sample answer to the question looks like.
                    
                                              
                    I DONT want to see any explanation or any additional lines or any other verbose in the output.
                    I also DONT need something like "Here is the answer:" etc in the output.

                    User_question:
                    {user_question}     
                                              """
                                                
                                                ''')

    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    response = chain.invoke({'user_question':query})

    return response