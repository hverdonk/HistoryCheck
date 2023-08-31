"""
@author Hannah Verdonk
"""
from langchain.document_loaders import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Loads files from a Github repo
# The Repository can be local on disk available at repo_path, or remote at clone_url that will be cloned to repo_path. 
# https://api.python.langchain.com/en/latest/document_loaders/langchain.document_loaders.git.GitLoader.html
loader = GitLoader('/home/hannah/Dropbox/Other Stories/To faraway worlds and back/FFF-chatbot-data')
data = loader.load()

# Split documents into chunks
# alternatively, split by token: https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/split_by_token
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 50,
    length_function = len,  # it's pretty common to pass a token counter here
    add_start_index = True,
)
chunks = text_splitter.split_documents(data)

# Embed chunks and store them in ChromaDB vector store (a local database of chunk vector embeddings)
model_name = "BAAI/bge-small-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
db = Chroma.from_documents(documents=chunks, embedding=hf)  # db loads in memory


# re-order docs after retrieval to avoid performance degradation:
# https://python.langchain.com/docs/modules/data_connection/document_transformers/post_retrieval/long_context_reorder

# prepend the following instruction to the user query
instruction = "Represent this sentence for searching relevant passages: "

