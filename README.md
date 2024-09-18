# ML_Assignment
# Level 1
- Documentation:
- a. Approach
Data Preparation: A subset of the 20 Newsgroups dataset was used, focusing on a single topic. The text was preprocessed by removing headers, footers, and unnecessary metadata to clean the dataset.
Model Integration: I used Gemini 1.5 flash model with a RAG approach, embedding documents and retrieving the most relevant ones before generating a response.
UI Development: Chainlit was used to create a simple web interface that allows users to interact with the system by asking questions and receiving answers.
- b. Challenges
Data Preprocessing: Removing irrelevant content from the dataset to retain meaningful text for LLM-based question answering was essential.
Retrieval Mechanism: The RAG approach requires a robust method for document retrieval. Fine-tuning and experimenting with different vector stores like Chroma helped improve the relevance of retrieved documents.
- c. Improvements for Future
Scalability: For larger datasets, consider adding more efficient retrieval mechanisms (e.g., FAISS).
Fine-Tuning LLM: With more time and resources, fine-tuning a LLM model on the 20 Newsgroups dataset could improve the model’s performance for domain-specific questions.
Enhancing the UI: Consider using more advanced front-end frameworks (like Streamlit or Flask) to improve the user interface.
