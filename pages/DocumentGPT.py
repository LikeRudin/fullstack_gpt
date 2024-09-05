import requests
import os

import streamlit as st


from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore


from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📜"
)


def get_file_path(middle_path, name):
    return f"./.cache/{middle_path}/{name}"


@st.cache_resource
def get_llm(api_key):
    return ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    openai_api_key=api_key,
)

def is_valid_api_key(api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
        return response.status_code ==200
    except requests.RequestException:
        return False


@st.cache_data(show_spinner="Embedding file💾...")
def embed(file, api_key):
    file_content = file.read()
    file_path = get_file_path("files", file.name);
    with open(file_path, "wb") as opened_file:
        opened_file.write(file_content)

    cache_dir = LocalFileStore(get_file_path("embeddings", file.name))
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        seperator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(documents, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def draw_message(message,role):
    with st.chat_message(role):
        st.markdown(message)

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message,role):
    draw_message(message,role)
    save_message(message,role)

def paint_history():
    for message in st.session_state["messages"]:
        draw_message(
            message["message"],
            message["role"],
        )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)




basic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful assistant here to assist the user.
        The user may ask questions and provide previous conversation history with you. 
        Your task is to answer the current question by leveraging both the provided context and the conversation history. 
        If you don’t know the answer, simply say you don’t know—please avoid making up information. 
        Use only the information from the context and chat history to formulate your response.
        """
    ),
    ("human", "Conversation history:\n{chat_history}\n\nCurrent question: {question}")
])

circuit_search_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            Use the following excerpt from a longer document to determine if any of the text is relevant for answering the question. 
            Return any relevant text verbatim. If there is no relevant text, return an empty string.
            -------
            {context}
            """,
        ),
        ("human", "{question}"),
])


st.title("Document GPT")

st.markdown(
    """
    Welcome! 
    Use this chatbot to ask question about document you uploaded
    
    Upload your files on the sidebar
    """
)

with st.sidebar:
    api_key = st.text_input("paste your OpenAI Api Key", type="password")
    file = st.file_uploader(
        "Upload a file. (extention: .txt .pdf .docx)",
        type=["pdf", "txt", "docx"],
    )

if api_key:
    if is_valid_api_key(api_key):
        os.environ["OPENAI_API_KEY"] = api_key 
        st.success("Api key is valid")
    if file:
        retriever = embed(file, api_key)
        llm = get_llm(api_key)
        draw_message("now is time to question. Ask anything","ai")
        paint_history()
        message = st.chat_input("Ask. What do you want to know?")
        if message:
            send_message(message, "human")

            repeat_chain = circuit_search_prompt | llm

        def process_document(document, question):
            result = repeat_chain.invoke({"context": document.page_content, "question": question})
            return result.content


        def map_docs(inputs):
            documents = inputs["documents"]
            question = inputs["question"]

            results = []
            for doc in documents:
                results.append(process_document(doc, question))
            return "\n\n".join(results)

        map_chain = {
            "documents": retriever,
            "question": RunnablePassthrough(),
            } | RunnableLambda(map_docs)
        final_chain = {
            "context": map_chain,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: st.cache_data
            } | basic_prompt | llm
        with st.chat_message("ai"):
            final_chain.invoke(message)


    


