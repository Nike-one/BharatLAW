from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import time

from new import api

st.set_page_config(page_title="LawGPT")

col1, col2, col3 = st.columns([1,6,1])
with col2:
    st.image("https://github.com/harshitv804/LawGPT/assets/100853494/ecff5d3c-f105-4ba2-a93a-500282f0bf00", width=700)

st.markdown(
    """
    <style>
    /* Button Styles */
    div.stButton > button {
        border: 2px solid #4CAF50; /* Green border */
        background-color: #4CAF50; /* Green background */
        color: white; /* White text */
        padding: 10px 24px; /* Some padding */
        cursor: pointer; /* Pointer/hand icon */
        border-radius: 8px; /* Rounded corners */
        font-size: 16px; /* Large font size */
    }

    div.stButton > button:hover {
        background-color: #45a049; /* Darker green background on hover */
    }

    div.stButton > button:active {
        background-color: #3e8e41; /* Even darker green background when clicked */
    }

    /* Input Field Styles */
    .stTextInput>div>div>input {
        border-radius: 20px !important;
        border: 1px solid #4CAF50 !important;
    }

    /* General App Styles */
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 2rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .css-1d391kg {padding-top: 0rem;} /* Adjust Streamlit's default padding at the top */

    /* Hide Streamlit's Fullscreen Button */
    button[title="View fullscreen"] {
        visibility: hidden;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Your Streamlit app's content goes here

def reset_conversation():
  st.session_state.messages = []
  st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history",return_messages=True)

embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1",model_kwargs={"trust_remote_code":True,"revision":"289f532e14dbbbd5a04753fa58739e9ba766f3c7"})
db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity",search_kwargs={"k": 4})

prompt_template = """
<s>[INST]
As a legal chatbot with expertise in the Indian Penal Code, your primary objective is to deliver precise, accurate, and succinct responses to user inquiries. Please adhere to these guidelines:
- Respond in a bullet-point format to ensure clarity and brevity.
- Directly and accurately address the user's question with relevant information.
- Avoid providing additional information beyond what is necessary to answer the question.
- Do not generate content unrelated to the user's current question.
- Utilize available information to respond to queries outside our direct knowledge base, focusing solely on the user's current question without referring back to chat history.
- Ensure that your responses are strictly relevant to the context provided and the specific question asked.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
- [Provide answers in bullet points]

</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question', 'chat_history'])

# You can also use other LLMs options from https://python.langchain.com/docs/integrations/llms. Here I have used TogetherAI API
TOGETHER_AI_API= os.environ['TOGETHER_API_KEY']
llm = Together(
    model="togethercomputer/StripedHyena-Nous-7B",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=api
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

input_prompt = st.chat_input("Say something")

if input_prompt:
    # Display user's message
    with st.chat_message("user"):
        st.markdown(f"**You:** {input_prompt}")

    # Append user's message to the session state
    st.session_state.messages.append({"role": "user", "content": input_prompt})

    # Simulate the assistant thinking before responding
    with st.chat_message("assistant"):
        with st.spinner("Thinking 💡..."):
            result = qa.invoke(input=input_prompt)
            message_placeholder = st.empty()

            # Initialize the response message
            full_response = "⚠️ **_Note: Information provided may be inaccurate._** \n\n\n"
            for chunk in result["answer"]:
                # Simulate typing by appending chunks of the response over time
                full_response += chunk
                time.sleep(0.02)  # Adjust the sleep time to control the "typing" speed
                message_placeholder.markdown(full_response + " ▌", unsafe_allow_html=True)

    # Append assistant's message to the session state
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

    # Display a visually appealing reset button
    if st.button('🗑️ Reset All Chat', on_click=reset_conversation):
        st.experimental_rerun()

# Apply custom styling to enhance the chat UI
st.markdown(
    """
    <style>
    /* Chat message styling */
    .stChatMessage {
        border-radius: 20px;
        padding: 10px;
    }
    /* Custom styling for user messages */
    .stChatMessage[data-testid="stChatMessage-user"] {
        background-color: #e1f5fe;
    }
    /* Custom styling for assistant messages */
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background-color: #ede7f6;
    }
    /* Style the reset button */
    div.stButton > button {
        border: none;
        border-radius: 20px;
        padding: 8px 24px;
        font-size: 16px;
        color: white;
        background-color: #6c757d;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)