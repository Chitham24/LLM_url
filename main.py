# import os
# import streamlit as st
# import pickle
# import time
# import google.generativeai as genai
# import langchain
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import FAISS

# from dotenv import load_dotenv
# load_dotenv()

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# st.title("URL Reader")
# st.sidebar.title("Enter News Article URLs")

# urls = []

# for i in range(3):
#     url = st.sidebar.text_input(f"URL {i+1}")
#     urls.append(url)

# process_url_clicked = st.sidebar.button("Process URLs")

# main_placefolder = st.empty()
# file_path = "vector_index.pkl"
# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


# if process_url_clicked:
#     #load urls
#     loader = UnstructuredURLLoader(urls = urls)
#     main_placefolder.text("Data Loading...")
#     data = loader.load()
#     #split data
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=['\n\n','\n','.',','],
#         chunk_size = 1000
#     )
#     main_placefolder.text("Splitting Text...")
#     docs = text_splitter.split_documents(data)

#     #embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#     vectorindex_genai = FAISS.from_documents(docs, embeddings)

#     main_placefolder.text("Building Embedding Vector...")
#     time.sleep(2)

#     # Extract components for pickling
#     faiss_index = vectorindex_genai.index
#     docstore = vectorindex_genai.docstore
#     index_to_docstore_id = vectorindex_genai.index_to_docstore_id

#     # Serialize components to a file
#     with open(file_path, "wb") as f:
#         pickle.dump((faiss_index, docstore, index_to_docstore_id), f)

# query = main_placefolder.text_input("Question: ")
# if query:
#     if os.path.exists(file_path):
#         # Load vector database from the pickle file
#         with open(file_path, "rb") as f:
#             loaded_faiss_index, loaded_docstore, loaded_index_to_docstore_id = pickle.load(f)
#         # Reconstruct the FAISS vector store
#         loaded_vectorindex = FAISS(
#             index=loaded_faiss_index,
#             docstore=loaded_docstore,
#             index_to_docstore_id=loaded_index_to_docstore_id,
#             embedding_function=embeddings.embed_query,  # Pass the embedding function here
#         )
#         chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=loaded_vectorindex.as_retriever())
#         result = chain({"question": query}, return_only_outputs=True)
#         #result --> {"answer":" ", "sources": [] }
#         st.subheader("Answer")
#         st.write(result["answer"])

#         st.subheader("Sources")
#         st.write(result["sources"])
     


# Cloudscrapper

import os
import time
import pickle
import streamlit as st
from urllib.parse import urlparse
import cloudscraper
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

st.title("URL Reader")
st.sidebar.title("Enter News Article URLs")

def is_valid_url(url):
    """Check if a given string is a valid URL."""
    parsed_url = urlparse(url)
    return all([parsed_url.scheme, parsed_url.netloc])

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if is_valid_url(url):
        urls.append(url)
    else:
        st.sidebar.warning(f"Invalid URL: {url}")

process_url_clicked = st.sidebar.button("Process URLs")

main_placefolder = st.empty()
file_path = "vector_index.pkl"
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Cloudscraper-based URL Loader
class CloudscraperURLLoader:
    def __init__(self, urls):
        self.urls = urls
        self.scraper = cloudscraper.create_scraper()

    def load(self):
        documents = []
        for url in self.urls:
            try:
                response = self.scraper.get(url)
                response.raise_for_status()  # Raise an error for bad responses
                content = response.content
                documents.append(Document(page_content=content.decode('utf-8'), metadata={"source": url}))
            except Exception as e:
                print(f"Error fetching {url}: {e}")
        return documents

if process_url_clicked:
    if not urls:
        st.warning("No valid URLs provided. Please enter valid URLs.")
    else:
        # Load URLs using the Cloudscraper loader
        loader = CloudscraperURLLoader(urls=urls)
        main_placefolder.text("Data Loading...")
        data = loader.load()

        # Check if data contains valid documents
        if isinstance(data, list) and all(hasattr(doc, 'page_content') for doc in data):
            # Split data
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placefolder.text("Splitting Text...")
            docs = text_splitter.split_documents(data)

            # Embeddings
            vectorindex_genai = FAISS.from_documents(docs, embeddings)

            main_placefolder.text("Building Embedding Vector...")
            time.sleep(2)

            # Extract components for pickling
            faiss_index = vectorindex_genai.index
            docstore = vectorindex_genai.docstore
            index_to_docstore_id = vectorindex_genai.index_to_docstore_id

            # Serialize components to a file
            with open(file_path, "wb") as f:
                pickle.dump((faiss_index, docstore, index_to_docstore_id), f)
        else:
            st.error("No valid documents to process.")

query = main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        # Load vector database from the pickle file
        with open(file_path, "rb") as f:
            loaded_faiss_index, loaded_docstore, loaded_index_to_docstore_id = pickle.load(f)
        # Reconstruct the FAISS vector store
        loaded_vectorindex = FAISS(
            index=loaded_faiss_index,
            docstore=loaded_docstore,
            index_to_docstore_id=loaded_index_to_docstore_id,
            embedding_function=embeddings.embed_query,  # Pass the embedding function here
        )
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=loaded_vectorindex.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        # result --> {"answer":" ", "sources": [] }
        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        st.write(result["sources"])





