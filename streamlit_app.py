from typing import Dict, Any
import asyncio

# Create a new event loop
loop = asyncio.new_event_loop()

# Set the event loop as the current event loop
asyncio.set_event_loop(loop)

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    PromptTemplate,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import streamlit as st
from streamlit_pills import pills


st.set_page_config(
    page_title=f"Chat avec Liste des médicaments RAMQ , powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

if "messages" not in st.session_state:  # Initialize the chat messages history
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Posez-moi des questions sur les médicaments"}
    ]

st.title(
    f"Chat avec Liste des médicaments RAMQ , powered by LlamaIndex 💬🦙"
)
st.info(
    "This example is powered by the **[DataHub](https://mckesson.ai/l/datahub)**. to retrieve and chat with your data via a Streamlit app.",
    icon="ℹ️",
)

def add_to_message_history(role, content):
    message = {"role": role, "content": str(content)}
    st.session_state["messages"].append(
        message
    )  # Add response to message history

@st.cache_resource
def load_index_data():
    # define embedding function
    Settings.embed_model = OllamaEmbedding(
        model_name="mxbai-embed-large",
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0},
    )
    # define llm model to interact with
    Settings.llm = Ollama(
        model="llama3.2",
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0},
    )

    #WikipediaReader = download_loader(
    #    "WikipediaReader", custom_path="local_dir"
    #)
    #loader = WikipediaReader()
    #docs = loader.load_data(pages=["Snowflake Inc."])
    #service_context = ServiceContext.from_defaults(
    #    llm=Ollama(model="llama3.2", temperature=0.5)
    #)
    #index = VectorStoreIndex.from_documents(
    #    docs, service_context=service_context
    #)
    vector_store = DuckDBVectorStore.from_local("./duckdb/knowledge_base")

    knowledge_base = VectorStoreIndex.from_vector_store(vector_store)
    return knowledge_base

index = load_index_data()

selected = pills(
    "Choose a question to get started or write your own below.",
    [
        "Quelle est la marge bénéficiaire?",
        "What company did Snowflake announce they would acquire in October 2023?",
        "What company did Snowflake acquire in March 2022?",
        "When did Snowflake IPO?",
    ],
    clearable=True,
    index=None,
)

if "chat_engine" not in st.session_state:  # Initialize the query engine
    st.session_state["chat_engine"] = index.as_chat_engine(
        chat_mode="context", verbose=True
    )

for message in st.session_state["messages"]:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# To avoid duplicated display of answered pill questions each rerun
if selected and selected not in st.session_state.get(
    "displayed_pill_questions", set()
):
    st.session_state.setdefault("displayed_pill_questions", set()).add(selected)
    with st.chat_message("user"):
        st.write(selected)
    with st.chat_message("assistant"):
        response = st.session_state["chat_engine"].stream_chat(selected)
        response_str = ""
        response_container = st.empty()
        for token in response.response_gen:
            response_str += token
            response_container.write(response_str)
        add_to_message_history("user", selected)
        add_to_message_history("assistant", response)

if prompt := st.chat_input(
    "Your question"
):  # Prompt for user input and save to chat history
    add_to_message_history("user", prompt)

    # Display the new question immediately after it is entered
    with st.chat_message("user"):
        st.write(prompt)

    # If last message is not from assistant, generate a new response
    # if st.session_state["messages"][-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response = st.session_state["chat_engine"].stream_chat(prompt)
        response_str = ""
        response_container = st.empty()
        for token in response.response_gen:
            response_str += token
            response_container.write(response_str)
        # st.write(response.response)
        add_to_message_history("assistant", response.response)

    # Save the state of the generator
    st.session_state["response_gen"] = response.response_gen
