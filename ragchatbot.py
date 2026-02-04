from itertools import chain

import pdfplumber
import streamlit as st
from click import prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama

st.header("My RAG Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("upload a pdf file and start asking questions", type="pdf" )

# extract contents from file and chunk it
if file is not None:
    #extract text from it
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    #st.write(text)

    #split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks =text_splitter.split_text(text)
    #st.write(chunks)

    #generating embeddings
    # embeddings = OpenAIEmbeddings(
    #     model = "text-embedding-3-small",
    #     openai_api_key = OPENAI_API_KEY
    # )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    #storing embeddings in vector db
    vector_store = FAISS.from_texts(chunks, embeddings)

    #get user question
    user_question = st.text_input("Type your question here")
    #generate answer
    # question -> embeddings -> similarity search -> llm -> response(CHAIN)
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    retriever = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs = {"k":4}
    )

    #define the llm and prompts
    # llm = ChatOpenAI(
    #     model = "gpt-40-mini",
    #     temperature = 0.3,
    #     max_tokens = 1000,
    #     openai_api_key = OPENAI_API_KEY
    # )

    llm = ChatOllama(
        model="mistral",  # or llama3 / phi3
        temperature=0.3
    )

    st.write("Testing LLM...")
    response = llm.invoke("Say hello in one sentence")
    st.write(response.content)

    #provide the prompts
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about a PDF document.\n\n"
         "Guidelines:\n"
         "1. Provide complete, well-explained answers using the context below.\n"
         "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
         "3. If the context mentions related information, include it to give fuller picture.\n"
         "4. Only use information from the provided context - do not use outside knowledge.\n"
         "5. Summarize long information, ideally in bullets where needed\n"
         "6. If the information is not in the context, say so politely.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])
    chain = (
        { "context" : retriever | format_docs, "question" : RunnablePassthrough()}
        |prompt
        |llm
        |StrOutputParser()
    )

    if user_question:
        response = chain.invoke(user_question)
        st.write(response)