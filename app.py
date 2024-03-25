import huggingface_hub
from huggingface_hub import hf_hub_download
import torch
from llama_cpp import Llama
import langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import params
import argparse
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA
import warnings
from typing import Any
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
#import tensorflow as tf


model_name = "TheBloke/Mistral-7B-OpenOrca-GGUF"
model_file = "mistral-7b-openorca.Q4_K_M.gguf"

model_path = hf_hub_download(model_name, filename=model_file)

model_kwargs = {
  "n_ctx":4096,    # Context length to use
  #"n_threads":4,   # Number of CPU threads to use
  "n_gpu_layers":500,# Number of model layers to offload to GPU. Set to 0 if only using CPU
  "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
  "n_batch":32,
  "max_tokens":500,
  "temperature":0

}

llm = Llama(model_path=model_path, **model_kwargs)

embeddings = OpenAIEmbeddings(openai_api_key = params.openai_api_key)


loaders = [
 WebBaseLoader("https://en.wikipedia.org/wiki/AT%26T"),
 WebBaseLoader("https://en.wikipedia.org/wiki/Bank_of_America")
]
data = []
for loader in loaders:
    data.extend(loader.load())

# Step 2: Transform (Split)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[
                                               "\n\n", "\n", "(?<=\. )", " "], length_function=len)
docs = text_splitter.split_documents(data)

client = MongoClient(params.mongodb_conn_string)
collection = client[params.db_name][params.collection_name]

# Reset w/out deleting the Search Index 
collection.delete_many({})

# Insert the documents in MongoDB Atlas with their embedding
# https://github.com/hwchase17/langchain/blob/master/langchain/vectorstores/mongodb_atlas.py
vectorstore = MongoDBAtlasVectorSearch.from_documents(
    docs, embeddings, collection=collection, index_name=params.index_name
)


client = MongoClient(params.mongodb_conn_string)
collection = client[params.db_name][params.collection_name]

# initialize vector store
vectorStore = MongoDBAtlasVectorSearch(
    collection, OpenAIEmbeddings(openai_api_key=params.openai_api_key), index_name=params.index_name
)

def query_data(queryy):
    retriever = vectorStore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm , chain_type = "stuff" , retriever = retriever)
    ret_output = qa.run(queryy)

    return ret_output

app = FastAPI( title="RAG_APIs",
    description="RAG APIs")

# This defines the data json format expected for the endpoint, change as needed
class TextInput(BaseModel):
    inputs: str
    parameters: dict[str, Any] | None

@app.get("/ok")
async def ok_endpoint():
    return {"message": "ok"}

@app.post("/generate/")
async def generate_text(data: TextInput) -> dict[str, str]:
    try:
        print(type(data))
        print(data)
        params = data.parameters or {}
        #response = llama2_model(prompt=data.inputs, **params)
        response = query_data(data.inputs)
        #model_out = response['choices'][0]['text']
        model_out= response
        return {"generated_text": model_out}
    except Exception as e:
        print(type(data))
        print(data)
        raise HTTPException(status_code=500, detail=len(str(e)))

