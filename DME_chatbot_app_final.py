import os
import json
from typing import Any
import nltk
import nltk.internals
import pytesseract
from collections import Counter
import uuid
import streamlit as st

from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import pdfplumber
import pickle

def get_pdf_chunks(path):    
    # path = "./data/2_06_dme_and_supplies-part-2.pdf"
    # elements = partition_pdf(filename=path,
    #                      strategy='hi_res',
    #                      infer_table_structure=False,
    #                      chunking_strategy='by_title',
    #                      max_characters=2000,
    #                      new_after_n_chars=1500,
    #                      starting_page_number=9) 
    elements = pickle.load(open("./data/Chunks/pdf_chunks_short_2000", "rb"))
    
    tables = []
    table_page_numbers=[]
    with pdfplumber.open(path) as pdf:
        for i in range (0, len(pdf.pages)):
            page = pdf.pages[i]
            page_num = page.page_number + 8
            x = len(tables)
            tables += page.extract_tables()
            y = len(tables)
            for i in range(0,(y-x)):
                table_page_numbers.append(page_num)
    
    from tabulate import tabulate
    table_elements = []
    for table in tables:
        table_html = tabulate(table, headers = 'firstrow', tablefmt="html")
        table_elements.append(table_html)
    return elements, table_page_numbers, table_elements

def get_table_text_elements(elements):
    class Element(BaseModel):
        type: str
        text: Any
        metadata: Any

    keys = ['page_number', 'filename']
    categorized_elements = []
    for element in elements:
        metadata=element.metadata.to_dict()
        metadata={key: metadata[key] for key in keys}
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element), metadata=metadata))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element), metadata=metadata))

    # table_elements = [e for e in categorized_elements if e.type == "table"]
    text_elements = [e for e in categorized_elements if e.type == "text"]

    return text_elements

def get_summaries(table_elements, text_elements):
    prompt_text = """You are an assistant tasked with summarizing text and tables.
    Give a concise summary of the provided table or text. Table or text chunk: {element}"""

    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = Ollama(model="llama3.1", verbose=True)
    summarize_chain = {"element":lambda x: x} | prompt | model | StrOutputParser()

    # table_summaries = summarize_chain.batch(table_elements, {"max_concurrency": 5})
    with open("./data/Summaries/table_summaries_short_2000", "r") as fp:
        table_summaries = json.load(fp)

    texts = [i.text for i in text_elements]
    # text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    with open("./data/Summaries/text_summaries_short_2000", "r") as fp:
        text_summaries = json.load(fp)

    return table_summaries, text_summaries, texts

def create_vectorstore(table_summaries, text_summaries, texts, table_page_numbers, table_elements, text_elements):
    vectorstore = Chroma(collection_name="summaries", embedding_function=OllamaEmbeddings(model='nomic-embed-text', show_progress=True))
    store = InMemoryByteStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(vectorstore=vectorstore,
                                 docstore=store,
                                 id_key=id_key,
                                 search_kwargs={'k':3})
    
    tables_docs = [Document(page_content=s, metadata={'page_number':t})
         for i,(s,t) in enumerate(zip(table_elements,table_page_numbers))]

    texts_docs = [Document(page_content=s.text, metadata=s.metadata)
         for i, s in enumerate(text_elements)]

    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(text_summaries)]

    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts_docs)))

    table_ids = [str(uuid.uuid4()) for _ in tables_docs]
    summary_tables=[
        Document(page_content=s, metadata={id_key: table_ids[i]})
        for i, s in enumerate(table_summaries)]

    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables_docs)))

    return retriever

def get_conversation_chain(retriever):
    general_system_template = r"""
    Answer the question based only on the given context, which can include both text and tables.
    Provide a detailed answer. If you don't know the answer, just say you don't know. 
    Provide section or table reference if possible. 
    ----
    {context}
    ----
    """
    general_user_template = "  'question':```{question}```"
    # doc_prompt = PromptTemplate(
    # template="Content: {page_content}\n Page:{page_number}", # look at the prompt does have page#
    # input_variables=["page_content","page_number"])
    messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    memory = ConversationBufferMemory(memory_key = 'chat_history', input_key='question', output_key='answer', return_messages = True)
    llm = Ollama(model="llama3.1")
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type = "stuff",
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs ={'prompt':qa_prompt}
    )
    return conversation_chain

def handle_userInput(user_question):

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    st.session_state.source_documents = response['source_documents']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    
    for i,docs in enumerate(st.session_state.source_documents):
        page_number = docs.metadata['page_number']
        st.write(f"**Source document {i+1}**: '{docs.page_content[0:100]}........{docs.page_content[-75:]}' on **Page No: {page_number}**", unsafe_allow_html=True)
                     
def main():
    st.set_page_config(page_title="Chat with PDF", page_icon=":books:")

    st.write(css, unsafe_allow_html = True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    st.header("Chat with PDF :books:")
    user_question = st.text_input("Ask a question about your document:")
    if user_question:
        handle_userInput(user_question)

    with st.sidebar:
        st.subheader("Your document")
        pdf_docs = st.file_uploader("Upload your PDF here and click on 'Process'")
        import tempfile
        if pdf_docs:
            temp_dir = tempfile.mkdtemp()
            newPath = os.path.join(temp_dir, pdf_docs.name)
            path = newPath.replace(os.sep, '/')
            with open(path, "wb") as f:
                f.write(pdf_docs.getvalue())

        if st.button("Process"):
            st.subheader("PDF chunks")
            with st.spinner("Creating Chunks of PDF"):
                
                elements, table_page_numbers, table_elements = get_pdf_chunks(path)

                text_elements= get_table_text_elements(elements)

                st.write(str(len(table_elements)) + " table chunks were extracted from the document")
                st.write(str(len(text_elements)) + " text chunks were extracted from the document")

            st.subheader("Summaries of Table and Text chunks")
            with st.spinner("Creating summaries of text and tables"):
                table_summaries, text_summaries, texts = get_summaries(table_elements, text_elements)
                st.write("Summaries of table and text chunks have been created")

            st.subheader("Embeddings and Vectorstore")
            with st.spinner("Creating embeddings and vectorstore"):
                retriever = create_vectorstore(table_summaries, text_summaries, texts, table_page_numbers, table_elements, text_elements)
                st.write("Finished creating embeddings and vectorstore. You can enter your question now.")
                
                st.session_state.conversation = get_conversation_chain(retriever)

if __name__ == '__main__':
    main()