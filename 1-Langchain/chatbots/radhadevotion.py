from dotenv import load_dotenv
import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEndpointEmbeddings
# load_dotenv()
# Streamlit UI Config
st.set_page_config(page_title="Radha Rani Bot", page_icon="🌸")
st.title("Premanand Ji Maharaj Devotional Motivation 🌸")

# 🔁 Take Hugging Face API token from user
hf_token = st.text_input("Enter your Hugging Face API Token", type="password")
st.write(hf_token)
if hf_token:
    try:
        # 🔁 Initialize embeddings with user-provided token
        model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        embeddings = HuggingFaceEndpointEmbeddings(
            model=model,
            huggingfacehub_api_token=hf_token
        )

        # 🔁 Load FAISS vector store
        base_path = os.path.dirname(__file__)
        index_path = os.path.join(base_path, "faiss1_index")
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_type="similarity", k=3)

        # 🔁 Load LLM (Groq)
        groq_api_key = os.getenv("GROQ_API_KEY")
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

        # 🔁 Prompt templates
        system_prompt = (
            "You are a devotional assistant that answers questions strictly based on the teachings and wisdom of Premanand Ji Maharaj. "
            "You must always reply in the same language as the user's question. If the user asks in Hinglish, reply in Hinglish. "
            "Always mention Goddess Radha as 'Radha Rani' in every response. "
            "If the question is unclear, give a metaphorical answer based on Radha Rani's wisdom. "
            "If nothing is found, reply with 'जय श्री राधे रानी।' "
            "Don't repeat sentences, and keep answers concise within 5 lines. "
            "Use the following pieces of retrieved context to answer. \n\n{context}"
        )

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # 🔁 RAG Chain
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # 🔁 Chat history store
        if "store" not in st.session_state:
            st.session_state.store = {}

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # 🔁 Image and Input
        st.markdown(
            """
            <div style="text-align: center;">
                <img src="https://m.media-amazon.com/images/I/81ikyIaQYNL.jpg" width="300" height="250">
                <p style="font-size:16px;">🌸 Radha Rani with Krishna</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        session_id = "user_123"
        input_text = st.text_input("Ask me anything:")

        if input_text:
            with st.spinner("Thinking..."):
                response = conversational_rag_chain.invoke(
                    {"input": input_text},
                    config={"configurable": {"session_id": session_id}},
                )
                st.markdown("### Marg Darshan:")
                st.write(response["answer"])

    except Exception as e:
        st.error(f"⚠️ Failed to load model or vectorstore: {e}")

else:
    st.warning("Please enter your Hugging Face API token to continue.")
