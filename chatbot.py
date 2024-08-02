import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = "sk-xxxxxxxxxxxxrBoUA" #OPENAI_KEY

#Upload PDF files
st.header("My First Chatbot !")

with st.sidebar: #left-side sidebar
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")


#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text+=page.extract_text()
        #st.write(text) made to write the text we read

#Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000, #1000 characters per chunk
        chunk_overlap=150, #bring the last 150 characters to understand context
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)

#Generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

#Creation Vector Store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

#Get User Question
    user_question = st.test_input("Type your question here")


#Do Similarity Search
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)

#Define the LLM
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature = 0, #low value to get the least randomness
        max_tokens = 1000,
        model_name = "gpt-3.5-turbo"
    )

#Output results 
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_document = match, question = user_question)
    st.write(response)
