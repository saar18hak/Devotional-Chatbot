import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_core.tracers import LangChainTracer

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"

# Initialize LangSmith tracer
tracer = LangChainTracer()

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Question: {question}")
])

# LLM and output parser
llm = Ollama(model="gemma:2b")
parser = StrOutputParser()

# Chain with LangSmith tracing
chain = (prompt | llm | parser).with_config({"callbacks": [tracer]})

# Streamlit UI
st.title("🔗 LangChain with Ollama and LangSmith")

input_text = st.text_input("Ask me anything:")

if input_text:
    with st.spinner("Thinking..."):
        response = chain.invoke({"question": input_text})
        st.markdown("### 🤖 Response:")
        st.write(response)
