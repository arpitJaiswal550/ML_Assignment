from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA 
from langchain.vectorstores import Chroma
import faiss
from langchain.vectorstores import FAISS
import chainlit as cl
import os
from dotenv import load_dotenv
load_dotenv()
import re
from sklearn.datasets import fetch_20newsgroups
import wikipediaapi
import warnings
warnings.filterwarnings('ignore')

def fetch_wikipedia_articles(topic_list):
    wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)','en')
    articles = []
    for topic in topic_list:
        page = wiki_wiki.page(topic)
        if page.exists():
            articles.append(page.text)
    return articles

# Example: Fetch articles related to 'Technology' and 'Healthcare'
wiki_topics = ['Healthcare', 'Medicine', 'Space', 'Electronics']
wikipedia_docs = fetch_wikipedia_articles(wiki_topics)
# print(wikipedia_docs)

# Select subset of documents from a specific category (e.g., 'rec.sport.hockey')
categories = ['sci.space','sci.med','sci.electronics']
dataset = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
# print(dataset.data)

# Preprocessing function to clean text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces/newlines
    return text

# Preprocess both document sets
newsgroup_documents = [preprocess_text(doc) for doc in dataset.data]
wikipedia_documents = [preprocess_text(doc) for doc in wikipedia_docs]

# Combine both sets into one list of documents
all_documents = newsgroup_documents + wikipedia_documents
# print(all_documents)

# Initialize the text splitter with a chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Number of characters per chunk
    chunk_overlap=200  # Number of overlapping characters between chunks
)

# Function to chunk the documents
def chunk_documents(documents):
    chunked_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc)
        chunked_docs.extend(chunks)
    return chunked_docs

# Apply chunking to all documents
chunked_documents = chunk_documents(all_documents)

# Load Gemini embeddings
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create embeddings for all documents
# document_embeddings = gemini_embeddings.embed_documents(chunked_documents)
# Use FAISS to create the vector store with chunked document embeddings
faiss_index = FAISS.from_texts(chunked_documents, gemini_embeddings)

# Save FAISS index to disk
faiss_index.save_local("faiss_index_dir")
