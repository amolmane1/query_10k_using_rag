#!/usr/bin/env python
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware

import os
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
SEC_API_KEY = os.environ.get("SEC_API_KEY")

print(OPENAI_API_KEY)
print(LANGCHAIN_API_KEY)
print(SEC_API_KEY)

import html2text
from sec_api import QueryApi, RenderApi
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel


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


def create_vectorstore_for_company(stock_symbol):
    print("creating new")
    queryApi = QueryApi(api_key=SEC_API_KEY)
    query = {
        "query": "ticker:{} AND formType:\"10-K\"".format(stock_symbol),
        "from": "0",
        "size": "1",
        "sort": [{ "filedAt": { "order": "desc" } }]
    }
    response = queryApi.get_filings(query)

    link_to_10k = response["filings"][0]["linkToFilingDetails"]
    print(link_to_10k)

    renderApi = RenderApi(api_key=SEC_API_KEY)

    target_10k = renderApi.get_filing(link_to_10k)
    target_10k_full_text = html2text.html2text(target_10k)
    print(len(target_10k_full_text))
    docs = [Document(page_content=target_10k_full_text)]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), persist_directory="./chroma_{}".format(stock_symbol))
    print("finished create_vectorstore_for_company")
    return vectorstore


def get_retriever(stock_symbol):
    stock_symbol = stock_symbol.upper()
    vectorstore_filename = "chroma_{}".format(stock_symbol)
    if vectorstore_filename in os.listdir("./"):
        vectorstore = Chroma(persist_directory=vectorstore_filename, embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    else:
        vectorstore = create_vectorstore_for_company(stock_symbol)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    print("finished get_retriever")
    return retriever

# vectorstore = Chroma(persist_directory="chroma_MSFT", embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retriever = get_retriever("MSFT")

template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
retriever_chain = RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
rag_chain = (
    retriever_chain
    | prompt
    | llm
    | StrOutputParser()
)


# @app.post("/load_10k")
# def read_item(stock_symbol: str):
#     _ = get_retriever(stock_symbol)
#     return "Started loading 10k"


@app.post("/query_10k")
def read_item(query: str):
    response = rag_chain.invoke(query)
    # print(response)
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)