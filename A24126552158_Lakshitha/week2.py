import streamlit as st
from google import genai
from dotenv import load_dotenv
import os


# PAGE CONFIG

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)


# LOAD API KEY

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in .env")
    st.stop()


# GEMINI CLIENT

client = genai.Client(api_key=API_KEY)


# SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = []

# UI
st.title("🤖 AI Chatbot")

st.caption("Algoverse")

st.divider()

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
prompt = st.chat_input("Type your message...")

if prompt:

    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant reply
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

                reply = response.text

            except Exception as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    reply = "⚠️ The AI server is currently overloaded due to high demand. Please try again later."

                reply = f"❌ Error:\n\n{str(e)}"

        st.markdown(reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })