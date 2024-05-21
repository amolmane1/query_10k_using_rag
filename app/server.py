#!/usr/bin/env python
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import html2text
from sec_api import QueryApi, RenderApi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents.base import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
SEC_API_KEY = os.environ.get("SEC_API_KEY")


app = FastAPI(
    title="Querying 10k's using RAG",
    version="1.0",
    description="A simple API to retrieve and store a company's 10k into a vector database, and then query it.",
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
    """
    Uses the SEC's Query API to get the link to the given company's latest 10k. Uses the SEC's Download API to download that 10k. Saves that 10k as a vectorstore.

    Parameters:
        stock_symbol (str): the ticker symbol of a company, upto 4 characters

    Returns:
        vectorstore (obj): a vectorstore of the company's 10k
    """
    queryApi = QueryApi(api_key=SEC_API_KEY)
    query = {
        "query": "ticker:{} AND formType:\"10-K\"".format(stock_symbol),
        "from": "0",
        "size": "1",
        "sort": [{ "filedAt": { "order": "desc" } }]
    }
    response = queryApi.get_filings(query)
    if len(response['filings']) == 0:
        raise ValueError("Stock symbol: {} does not exist".format(stock_symbol))
 
    link_to_10k = response["filings"][0]["linkToFilingDetails"]

    renderApi = RenderApi(api_key=SEC_API_KEY)
    target_10k = renderApi.get_filing(link_to_10k)

    # remove the html tags from the downloaded 10k so that it's more readable. also to reduce the number of tokens.
    target_10k_full_text = html2text.html2text(target_10k)
    docs = [Document(page_content=target_10k_full_text)]
    # split the cleaned 10k into snippets of 1000 characters each. each snippet becomes a "document" in the vectorstore, and is compared to the query during the retrieval step.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    # create a vectorstore and store it
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), persist_directory="./chroma_{}".format(stock_symbol))
    return vectorstore


def get_retriever(stock_symbol):
    """
    Checks if a vecstorstore has already been created for a given stock symbol. If yes, it loads and returns the retriever for that. If no, it creates one.
    
    Parameters:
        stock_symbol (str): the ticker symbol of a company, upto 4 characters
    Returns:
        retriever (obj): retriever object for that company's vectorstore, to be used in the RAG chain
    """
    vectorstore_filename = "chroma_{}".format(stock_symbol)
    # If a vectorstore has already been created, load it. If no, create one
    if vectorstore_filename in os.listdir("./"):
        vectorstore = Chroma(persist_directory=vectorstore_filename, embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    else:
        vectorstore = create_vectorstore_for_company(stock_symbol)
    # set the retriever to return the 3 most similar snippets to the query
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever


def format_docs(docs):
    """
    Formats the most relevant documents to a query for use in an LLM prompt.
    """
    return "\n\n".join(doc.page_content for doc in docs)


template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
retriever = None


@app.post("/load_10k")
def load_10k(stock_symbol: str):
    """
    Set the 10k/vectorstore/retriever to be used for querying.
    """
    try:
        stock_symbol = stock_symbol.upper()
        global retriever
        retriever = get_retriever(stock_symbol)
        return "Loaded 10k for {}".format(stock_symbol)
    except ValueError as err:
        return "Unable to load 10k for {0}: {1}".format(stock_symbol, err)
    except:
        return "Unable to load 10k for {}".format(stock_symbol)


@app.post("/query_10k")
def query_10k(query: str):
    """
    Takes a question about the loaded 10k, and runs RAG to get the answer.
    """
    try:
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = rag_chain.invoke(query)
        return response
    except:
        return "Unable to query 10k"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)