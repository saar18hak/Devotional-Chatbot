#import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers import LangChainTracer
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()
#os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
#os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
#os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
#os.environ["LANGCHAIN_TRACKING_V2"] = "true"

# Load and split documents
loader = TextLoader("1-Langchain/chatbots/devotion.txt", encoding="utf-8")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs)

# Embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = FAISS.from_documents(final_documents, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", k=3)

# LLM from Groq
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt Templates
system_prompt = (
    "You are a devotional assistant that answers questions strictly based on the teachings and wisdom of Premanand Ji Maharaj. " 
    "You must always reply in the same language as the user's question. If the user asks in Hinglish (a mix of Hindi and English), reply in Hinglish. "
    "Always mention Goddess Radha as 'Radha Rani' in every response. "
    "If the question is unclear, give a metaphorical answer based on Radha Rani's wisdom. "
    "If nothing is found, reply with 'जय श्री राधे रानी।' "
    "Don't repeat sentences, and keep answers concise within 5 lines."
    "Use the following pieces of retrieved context to answer. \n\n"
    "{context}"
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Tracer (Optional, for LangSmith logging)
tracer = LangChainTracer()

# RAG + History-aware chain setup
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain).with_config({"callbacks": [tracer]})

# In-memory session store
# store = {}
# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]
#import streamlit as st

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


# Final conversational chain with memory
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# Streamlit UI
st.set_page_config(page_title="Radha Rani Bot", page_icon="🌸")
st.title("Premanand Ji Maharaj Devotional Motivation 🌸")

st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://m.media-amazon.com/images/I/81ikyIaQYNL.jpg" width="300" height="250">
        <p style="font-size:16px;">🌸 Radha Rani with Krishna</p>
    </div>
    """,
    unsafe_allow_html=True
)

session_id = "user_123"  # Can be made dynamic if needed
input_text = st.text_input("Ask me anything:")

# if input_text:
#     with st.spinner("Thinking..."):
#         response = conversational_rag_chain.invoke(
#             {"input": input_text},
#             config={"configurable": {"session_id": session_id}},
#         )
#         st.markdown("### 🤖 Response:")
#         st.write(response['answer'])

import requests  # Add this at the top if not already imported

# Inside Streamlit UI code
if input_text:
    with st.spinner("Thinking..."):
        response = conversational_rag_chain.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": session_id}},
        )

        # Extract response text
        response_text = response['answer']

        # Display response text
        st.markdown("### Marg Darshan:")
        st.write(response_text)

        # ---- ElevenLabs Voice Generation ----
        elevenlabs_api_key = "sk_04d47fc7e33d48f0a041b102e4d3af645089115159fd69ab"  # Replace with your actual key
        voice_id = "ttpam6l3Fgkia7uX33b6"  # Replace with actual voice ID

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": elevenlabs_api_key,
            "Content-Type": "application/json"
        }
        data = {
            "text": response_text,
            "voice_settings": {"stability": 0.4, "similarity_boost": 0.75}
        }

        tts_response = requests.post(url, headers=headers, json=data)

        if tts_response.status_code == 200:
            with open("output.mp3", "wb") as f:
                f.write(tts_response.content)
            st.audio("output.mp3", format="audio/mp3")
        else:
            st.error("❌ Failed to generate voice. Check ElevenLabs API key or Voice ID.")


print("store:",st.session_state.store)
# Show stored chat messages for debugging
history = get_session_history("user_123")
for msg in history.messages:
    print(f"{msg.type.upper()}: {msg.content}")
