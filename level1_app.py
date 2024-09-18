import warnings
warnings.filterwarnings('ignore')
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA 
from langchain.vectorstores import Chroma
import chainlit as cl
import os
from dotenv import load_dotenv
load_dotenv()
import re
from sklearn.datasets import fetch_20newsgroups

# Select subset of documents from a specific category (e.g., 'rec.sport.hockey')
categories = ['rec.sport.hockey']
dataset = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))

# Preprocessing the text documents (Remove headers, footers, signatures)
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove unwanted characters
    return text

documents = [preprocess_text(doc) for doc in dataset.data]

# Load Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  

# Create a document search index using the embeddings
doc_search = Chroma.from_texts(documents, embeddings)


# Load the Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                           verbose=True,
                           temperature=0,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))

# Define the RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=doc_search.as_retriever())

# Function to answer questions
def answer_question(query):
    response = qa_chain.invoke(query)
    return str(response['result'])

# # Test with a sample question
# query = "Who won the last hockey game?"
# answer = answer_question(query)
# print(answer)



# Chainlit function to handle user messages
@cl.on_message
async def main(message: str):
    # Debugging: Print the message to see its contents
    message = message.content
    print(f"Received message: {message}")
    # Ensure the message is a string and not None
    if not isinstance(message, str):
        await cl.Message(content="Invalid input. Please enter a valid question.").send()
        return
    
    # Answer the question using the QA system based on Gemini LLM
    try:
        answer = answer_question(message)
        if answer:
            await cl.Message(content=answer).send()
        else:
            await cl.Message(content="I couldn't find an answer to that question.").send()
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()

