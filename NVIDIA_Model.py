import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

load_dotenv()

os.environ['NVIDIA_API_kEY']=os.getenv('NVIDIA_API_KEY')

llm=ChatNVIDIA(model='meta/llama3-70b-instruct',tempetature=0.7)


def vector_embeddings():
    if 'vector' not in st.session_state:
        st.session_state.embeddings=NVIDIAEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader(os.getenv('path1'))
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=70)
        st.session_state.final_document=st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vector=FAISS.from_documents(st.session_state.final_document,st.session_state.embeddings)


prompt=ChatPromptTemplate.from_template(
"""
Answer the queations based on the provided context only.
Please provide the most accurate answers only.
<context>
{context}
</context>
Questions: {input}
"""
)
prompt1=st.text_input('Enter your text from Documents')

if st.button('Generate the answer'):
    st.write('Answer will generated in short')
    vector_embeddings()
    
    if prompt1:
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever=st.session_state.vector.as_retriever()
        retrieval_chain=create_retrieval_chain(retriever,document_chain)
        start=time.process_time()
        response=retrieval_chain.invoke({'input':prompt1})
        print('response time',time.process_time()-start)
        st.write(response['answer'])
        with st.expander('Document similariy search'):
            # find the relevent chunks 
            for i, doc in enumerate(response['context']):
                st.write('doc',doc.page_content)
                st.write('--------------------------------------')

