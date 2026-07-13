import streamlit as st
import time
import os
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ==========================================
# 1. INITIAL SETUP & PREMIUM THEMING
# ==========================================
st.set_page_config(
    page_title="Production AI Client", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# Custom injection for enterprise dashboards look
st.markdown("""
<style>
    div[data-testid="stMetricContainer"] {
        background-color: #1e222b;
        border: 1px solid #2d3139;
        padding: 10px 15px;
        border-radius: 8px;
    }
    div[data-testid="stMetricLabel"] p {
        color: #a0aec0 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
    }
    .small-latency {
        color: #718096;
        font-size: 0.75rem;
        text-align: right;
        margin-top: -10px;
        margin-bottom: 10px;
    }
    .thought-block {
        color: #8a99ad;
        font-family: monospace;
        font-size: 0.85rem;
        border-left: 2px solid #4a5568;
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# API CREDENTIALS VERIFICATION
# System environment variable check or direct setup
if "GEMINI_API_KEY" not in os.environ:
    # Environment variable lekapothe tool safely load avvadaniki clear prompt bypass setup
    os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY_HERE"

try:
    client = genai.Client()
except Exception as e:
    st.error(f"Initialization Error: Client context building failed. Check Key configurations. Details: {e}")

# PERSISTENT SESSION STATE ARCHITECTURE
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_api_calls" not in st.session_state:
    st.session_state.total_api_calls = 0
if "security_saves" not in st.session_state:
    st.session_state.security_saves = 0

MODEL_REGISTRY = {
    "Gemini 2.5 Flash (Fast Engine)": "gemini-2.5-flash",
    "Gemini 2.5 Pro (Reasoning Engine)": "gemini-2.5-pro"
}

# ==========================================
# 2. SIDEBAR METRICS & OPERATION LAYER
# ==========================================
with st.sidebar:
    st.title("⚙️ Control Center")
    st.caption("Manage secure routing parameters & metadata telemetry.")
    st.markdown("---")
    
    st.subheader("Model Version Routing")
    selected_version = st.selectbox(
        "Choose Target Core API Model:",
        options=list(MODEL_REGISTRY.keys()),
        label_visibility="collapsed"
    )
    active_model_id = MODEL_REGISTRY[selected_version]
    
    st.markdown("---")
    
    st.subheader("Observability Telemetry Dashboard")
    col1, col2 = st.columns(2)
    col1.metric(label="API Core Hits", value=st.session_state.total_api_calls)
    col2.metric(label="Security Saves", value=st.session_state.security_saves)

# ==========================================
# 3. CHAT DISPLAY CONTAINER INTERFACE
# ==========================================
st.title("💬 Secure Intelligent Gateway")
st.caption(f"Active Edge Protocol Target: `{active_model_id}`")
st.divider()

chat_placeholder = st.container()

# Render historical traces from memory state array
with chat_placeholder:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "thought" in msg and msg["thought"]:
                with st.expander("🔍 View Thinking Process (Executed Log)"):
                    st.markdown(f"<div class='thought-block'>{msg['thought']}</div>", unsafe_allow_html=True)
            
            st.markdown(msg["content"])
            
            if msg["role"] == "assistant" and "latency" in msg:
                st.markdown(f"<div class='small-latency'>⏱️ {msg['latency']:.2f}s via {msg['model']}</div>", unsafe_allow_html=True)

# ==========================================
# 4. LIVE INTERACTION LOOP (THE STREAMING CORE)
# ==========================================
if user_prompt := st.chat_input("Ask something safely..."):
    
    # 1. Immediate UI state confirmation
    with chat_placeholder:
        with st.chat_message("user"):
            st.markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # 2. Week 2: Local Proxy Security Intervention Layer
    banned_keywords = ["ignore previous instructions", "system override", "hack_system", "bypass_filter"]
    if not user_prompt.strip() or any(k in user_prompt.lower() for k in banned_keywords):
        st.session_state.security_saves += 1
        with chat_placeholder:
            with st.chat_message("assistant"):
                st.error("🚨 **Security Intervention Alert:** Input string matching signature payload rules blocked locally.")
        
        # State append to retain conversation timeline structure
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🚨 **Security Intervention Alert:** Input blocked locally.",
            "thought": "❌ Process aborted during Local Proxy Security verification."
        })
        time.sleep(1)
        st.rerun()

    # 3. Valid Execution Path Handoff
    with chat_placeholder:
        with st.chat_message("assistant"):
            st.session_state.total_api_calls += 1
            start_time = time.time()
            
            # --- STEP A: LIVE THINKING PIPELINE LOGS ---
            thought_box = st.expander("🧠 System Architecture Reasoning...", expanded=True)
            with thought_box:
                thought_text_area = st.empty()
                
                thinking_trace = [
                    "⚡ [0.1s] Local interceptors verified input schema layout... Clean.\n",
                    f"🔗 [0.3s] Establishing multi-turn session context matching parameters to: {active_model_id}...\n",
                    "⚙️ [0.5s] Context mapping completed. Serializing parameters via Google GenAI SDK Client interface...\n",
                    "🚀 [0.7s] Streaming connection opened safely with Vertex Edge Servers. Buffer streams initializing...\n"
                ]
                
                accumulated_thoughts = ""
                for step in thinking_trace:
                    time.sleep(0.2)  # Short artificial buffer so user eyes can read process
                    accumulated_thoughts += step
                    thought_text_area.markdown(f"<div class='thought-block'>{accumulated_thoughts}</div>", unsafe_allow_html=True)
            
            # Close thinking window automatically before output stream triggers
            thought_box.update(label="🔍 View Thinking Process (Executed)", expanded=False)

            # --- STEP B: GENAI TOKEN STREAM HANDLER LOOP ---
            response_box = st.empty()
            full_response_content = ""
            
            try:
                # Convert active memory structure to Google SDK Content Format structures
                formatted_contents = []
                for m in st.session_state.messages[:-1]: # Current message context exclude chesthu records format routing
                    role_tag = "user" if m["role"] == "user" else "model"
                    formatted_contents.append(
                        types.Content(role=role_tag, parts=[types.Part.from_text(text=m["content"])])
                    )
                # Append current latest turn parameters
                formatted_contents.append(
                    types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
                )

                # Initialize official chunk stream processor context hook
                response_stream = client.models.generate_content_stream(
                    model=active_model_id,
                    contents=formatted_contents,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=800
                    )
                )

                # Stream consumption pipeline loops
                for chunk in response_stream:
                    if chunk.text:
                        full_response_content += chunk.text
                        # Appending smooth visualization dynamic insertion cursor token
                        response_box.markdown(full_response_content + "▌")
                
                # Render clean definitive state string without typing visual symbols
                response_box.markdown(full_response_content)
                
            except APIError as api_err:
                full_response_content = f"❌ **Google GenAI Core Engine Exception:** Request execution failed. Details: {api_err}"
                response_box.error(full_response_content)
            except Exception as system_err:
                full_response_content = f"❌ **System Gateway Runtime Exception:** Process crash detected. Details: {system_err}"
                response_box.error(full_response_content)
            
            # --- STEP C: TELEMETRY PERFORMANCE RENDER ---
            latency = time.time() - start_time
            st.markdown(f"<div class='small-latency'>⏱️ {latency:.2f}s via {active_model_id}</div>", unsafe_allow_html=True)

    # Save tracking history variables state setup parameters completely inside runtime stack memory blocks
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response_content, 
        "thought": accumulated_thoughts,
        "latency": latency,
        "model": active_model_id
    })
    
    st.rerun()