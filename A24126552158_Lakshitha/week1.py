import streamlit as st
from google import genai
from dotenv import load_dotenv
import os
import string

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="AI Chatbot with Data Audit",
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
# WEEK 2: DATA QUALITY AUDIT FUNCTIONS (The Validation Layer)
# =====================================================================

def pre_processing_audit(text: str) -> tuple[bool, str]:
    """
    Audits incoming user input BEFORE it is sent to the Gemini API.
    Returns: (is_clean, result_or_error_message)
    """
    cleaned_text = text.strip()
    
    # Rule 1: Empty Input Check
    if not cleaned_text:
        return False, "⚠️ Audit Warning: Input is empty. Please enter a valid message."
        
    # Rule 2: Garbage Input / Pure Punctuation Check (e.g., "???")
    if all(char in string.punctuation for char in cleaned_text):
        return False, "⚠️ Audit Warning: Input contains only symbols. Please enter meaningful text."
        
    # Rule 3: Security / Prompt Injection Check
    banned_keywords = ["ignore instructions", "system override", "hack_system", "badword1"]
    for word in banned_keywords:
        if word in cleaned_text.lower():
            return False, f"⚠️ Security Audit Failed: Flagged keyword detected ('{word}'). This request is blocked."
            
    # Rule 4: Data Length/Token Spam Constraint
    if len(cleaned_text) > 1500:
        return False, "⚠️ Audit Warning: Input exceeds the maximum safety limit of 1500 characters."

    # If all rules pass, return a green light and the cleaned text
    return True, cleaned_text


def post_processing_audit(text: str) -> tuple[bool, str]:
    """
    Audits the generated AI response BEFORE it is displayed to the user.
    """
    # Rule 1: Empty Response Check
    if not text or len(text.strip()) == 0:
        return False, "⚠️ System Audit Error: The AI model generated an empty response. Please retry."
        
    # Rule 2: Structural Integrity Check
    if "internal failure" in text.lower() or "runtime error" in text.lower():
        return False, "⚠️ System Audit Error: The model output looks broken or corrupted."
        
    return True, text

# =====================================================================
# SESSION STATE & CHAT UI LOGIC
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 AI Chatbot with Data Audit")
st.caption("Algoverse — Week 2 Task: Data Quality Audit Implementation")
st.divider()

# Render chat history from session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input Box
prompt = st.chat_input("Type your message here...")

if prompt:
    # -----------------------------------------------------------------
    # STEP 1: EXECUTE PRE-PROCESSING AUDIT
    # -----------------------------------------------------------------
    is_input_clean, input_result = pre_processing_audit(prompt)
    
    # Always display the user's message on the screen first
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # If the input fails the audit, intercept execution locally and STOP
    if not is_input_clean:
        with st.chat_message("assistant"):
            st.error(input_result)
        st.session_state.messages.append({"role": "assistant", "content": input_result})
        st.stop()  # Stops Streamlit from running any further steps

    # -----------------------------------------------------------------
    # STEP 2: GEMINI API CALL (Only triggered if input is clean)
    # -----------------------------------------------------------------
    with st.chat_message("assistant"):
        with st.spinner("Auditing & Thinking..."):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=input_result  # Pass the audited and cleaned text
                )
                raw_reply = response.text
                
                # -------------------------------------------------------------
                # STEP 3: EXECUTE POST-PROCESSING AUDIT
                # -------------------------------------------------------------
                is_output_clean, final_reply = post_processing_audit(raw_reply)
                
                if not is_output_clean:
                    st.warning(final_reply)  # Display controlled audit warning
                else:
                    st.markdown(final_reply)  # Display standard clean response

            except Exception as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    final_reply = "⚠️ The AI server is currently overloaded. Please try again later."
                else:
                    final_reply = f"❌ Technical Error:\n\n{str(e)}"
                st.markdown(final_reply)

    # Save the final audited response to the session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_reply
    })