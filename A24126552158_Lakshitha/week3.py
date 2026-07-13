import streamlit as st
from google import genai
from dotenv import load_dotenv
import os
import string

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="AI Chatbot: Model Versioning",
    page_icon="🤖",
    layout="centered"
)

# =====================================================================
# LOAD API KEY & INITIALIZE GEMINI CLIENT
# =====================================================================
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in .env")
    st.stop()

client = genai.Client(api_key=API_KEY)

# =====================================================================
# WEEK 3: MODEL VERSIONING REGISTRY (The Control Layer)
# =====================================================================
# We define our verified models in a central registry dictionary.
# This completely eliminates hardcoding.
MODEL_REGISTRY = {
    "⚡ Gemini 2.5 Flash (Fast & Cost-Efficient)": "gemini-2.5-flash",
    "🧠 Gemini 2.5 Pro (Advanced Logic & Coding)": "gemini-2.5-pro"
}

# Add an interactive control center in the Streamlit Sidebar
st.sidebar.title("⚙️ Engine Configuration")
st.sidebar.caption("Week 3: Model Versioning Controller")

# User or developer selects the version from the dropdown menu
selected_display_name = st.sidebar.selectbox(
    "Select AI Model Version:",
    options=list(MODEL_REGISTRY.keys())
)

# Resolve the user friendly display name to the technical API string
ACTIVE_MODEL_VERSION = MODEL_REGISTRY[selected_display_name]

# Visual confirmation badge in the sidebar
st.sidebar.success(f"Active Engine:\n`{ACTIVE_MODEL_VERSION}`")

# =====================================================================
# WEEK 2: DATA QUALITY AUDIT FUNCTIONS (The Filtering Layer)
# =====================================================================
def pre_processing_audit(text: str) -> tuple[bool, str]:
    """Audits incoming user input BEFORE it hits the API."""
    cleaned_text = text.strip()
    
    if not cleaned_text:
        return False, "⚠️ Audit Warning: Input is empty. Please enter a valid message."
        
    if all(char in string.punctuation for char in cleaned_text):
        return False, "⚠️ Audit Warning: Input contains only symbols. Please enter meaningful text."
        
    banned_keywords = ["ignore instructions", "system override", "hack_system"]
    for word in banned_keywords:
        if word in cleaned_text.lower():
            return False, f"⚠️ Security Audit Failed: Flagged keyword detected ('{word}'). This request is blocked."
            
    if len(cleaned_text) > 1500:
        return False, "⚠️ Audit Warning: Input exceeds the safety limit of 1500 characters."

    return True, cleaned_text


def post_processing_audit(text: str) -> tuple[bool, str]:
    """Audits generated AI responses BEFORE they hit the screen."""
    if not text or len(text.strip()) == 0:
        return False, "⚠️ System Audit Error: The AI model generated an empty response."
    return True, text

# =====================================================================
# SESSION STATE & CHAT UI LOGIC
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 AI Chatbot Engine Dashboard")
st.caption("Algoverse — Week 3 Task: Model Versioning & Dynamic Routing")
st.divider()

# Render chat history with tracking metadata
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If metadata exists, show which engine version generated that response
        if "model_used" in msg:
            st.caption(f"Generated via: `{msg['model_used']}`")

# Chat Input Box
prompt = st.chat_input("Type your question here...")

if prompt:
    # 1. RUN PRE-PROCESSING AUDIT (Week 2 Layer)
    is_input_clean, input_result = pre_processing_audit(prompt)
    
    # Display user query instantly
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Intercept and stop if Week 2 audit fails
    if not is_input_clean:
        with st.chat_message("assistant"):
            st.error(input_result)
        st.session_state.messages.append({"role": "assistant", "content": input_result})
        st.stop()

    # 2. DYNAMIC API ROUTING (Week 3 Layer)
    with st.chat_message("assistant"):
        with st.spinner(f"Routing to {ACTIVE_MODEL_VERSION}..."):
            try:
                # The model variable is passed dynamically from the sidebar selection
                response = client.models.generate_content(
                    model=ACTIVE_MODEL_VERSION,  # <--- Dynamic variable!
                    contents=input_result
                )
                raw_reply = response.text
                
                # 3. RUN POST-PROCESSING AUDIT (Week 2 Layer)
                is_output_clean, final_reply = post_processing_audit(raw_reply)
                
                if not is_output_clean:
                    st.warning(final_reply)
                else:
                    st.markdown(final_reply)

            except Exception as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    final_reply = "⚠️ The selected model engine is currently overloaded. Please try again later."
                else:
                    final_reply = f"❌ Technical Error:\n\n{str(e)}"
                st.markdown(final_reply)

    # Save final message along with its versioning metadata tracker
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_reply,
        "model_used": ACTIVE_MODEL_VERSION  # Track version history metadata
    })