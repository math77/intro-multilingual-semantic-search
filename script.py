from dotenv import load_dotenv
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Cohere
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

import textwrap as tr
import random
import os

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")  

"""
Multilingual search semantic -> contextual search

"""

# chunk documents in smaller pieces
loader = TextLoader("steve-jobs-commencement.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


"""
  convert each document into a text embedding and store in a vector database
"""

# define the embedding model
embeddings = CohereEmbeddings(model="multilingual-22-12")

db = Chroma.from_documents(
  docs,
  embeddings
)

questions = [
  "What did the author liken The Whole Earth Catalog to?",
  "What was Reed College great at?",
  "What was the author diagnosed with?",
  "What is the key lesson from this article?",
  "What did the article say about Michael Jackson?",
]

prompt_template =  """Text: {context}

Question: {question}

Answer the question based on the text provided. If the text doesn't contain the answer, reply that the answer is not available."""

PROMPT = PromptTemplate(
  template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(
  llm=Cohere(model="command-nightly", temperature=0),
  chain_type="stuff",
  retriever=db.as_retriever(),
  chain_type_kwargs=chain_type_kwargs,
  return_source_documents=True
)

for question in questions:
  answer = qa({"query": question})

  print(answer["query"])
  print(answer["result"])
  print("--------------")