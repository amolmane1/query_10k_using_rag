#!/usr/bin/env python
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware

# import os
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# SEC_API_KEY = os.environ.get("SEC_API_KEY")

# print(OPENAI_API_KEY)
# print(LANGCHAIN_API_KEY)
# print(SEC_API_KEY)

import html2text
import json 
from sec_api import QueryApi, RenderApi
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_openai import ChatOpenAI


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# add_routes(
#     app,
#     ChatOpenAI(),
#     path="/openai",
# )

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    prompt | model,
    path="/joke",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)