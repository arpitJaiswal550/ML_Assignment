from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
import faiss
import chainlit as cl
import os
from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings('ignore')

# Load Gemini embeddings
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# Load the Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                           verbose=True,
                           temperature=0,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))


# Load the FAISS index from disk
faiss_index = FAISS.load_local("faiss_index_dir", gemini_embeddings, allow_dangerous_deserialization=True)

# Define the QA chain with FAISS retriever
def answer_question(query):
    retriever = faiss_index.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    # Get the response
    response = qa_chain.invoke(query)
    return response['result']

# # Test with a sample question
# query = "what is the side effects of taking prednisone?"
# answer = answer_question(query)
# print(answer)

# Chainlit function to handle user messages
@cl.on_message
async def main(message: str):
    # Debugging: Print the message to see its contents
    message = message.content
    # print(f"Received message: {message}")
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