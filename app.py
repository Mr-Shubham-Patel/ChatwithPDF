import streamlit as st
from PyPDF2 import PdfReader  ## reading pdf files
from langchain.text_splitter import RecursiveCharacterTextSplitter  ## split text into smaller chunks
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings  ## for generating embeddings using Google AI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS  ## Facebook AI Similarity Search 
from langchain_google_genai import ChatGoogleGenerativeAI  ## for chatting capabilities usig Googlen gen ai
from langchain.chains.question_answering import load_qa_chain  ## loading question-answering chain
from langchain.prompts import PromptTemplate    ## create prompt template
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    ## split the text chunks of 10000 characters with 1000 charcaters overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to
    provide all the details, if the answer is not in the provided context then say
    "answer is not available in the pdf. Don't provide the worng answer\n\n
    Context:\n {context}?"\n
    Question: \n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite",temperature=0.3)
    prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain = load_qa_chain(model,prompt=prompt)
    return chain

def user_input(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_data = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs = new_data.similarity_search(question)
    chain = get_conversational_chain()
    reponse = chain({"input_documents":docs,"question":question},return_only_outputs=True)
    st.write(reponse['output_text'])

st.set_page_config(page_title="Pdf Chat")
st.header("Chat with Pdf Web Application")
user_question = st.text_input("Ask a question from PDF")
if user_question:
    user_input(user_question)
    
with st.sidebar:
    st.title("Menu")
    pdf_docs = st.file_uploader("Upload your pdf files and click on submit button to process",
                                accept_multiple_files=True,type=["pdf"])
    submit = st.button("Submit & Process")
    if submit:
        with st.spinner("Processing...."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")