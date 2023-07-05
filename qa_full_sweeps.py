import os
import re
import wandb
import numexpr
import pandas as pd
from typing import List
from pydantic import BaseModel, Field, validator

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import QAGenerationChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

from langchain.callbacks import get_openai_callback

from dotenv import load_dotenv
load_dotenv("/Users/ayushthakur/integrations/llm-eval/apis.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

data_pdf = "data/qa/2304.12210.pdf"

# Build question-answer pairs for evaluation
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, 
    chunk_overlap  = 100,
    length_function = len,
)

loader = PyPDFLoader(data_pdf)
qa_chunks = loader.load_and_split(text_splitter=text_splitter)
print("Number of chunks for building qa eval set:", len(qa_chunks))




