import streamlit as st
from search import DSASearchEngine
from queryprocessor import QueryProcessor
from intent_detector import IntentDetector

# ----------------------------
# CONFIG
# ----------------------------

st.set_page_config(
    page_title="DSA AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------
# LOAD
# ----------------------------

engine = DSASearchEngine("data/dsa.json")
processor = QueryProcessor()
detector = IntentDetector()

# ----------------------------
# CUSTOM CSS
# ----------------------------

st.markdown("""
<style>

.main{
    background:#0E1117;
}

.stTextInput>div>div>input{
    border-radius:12px;
}

.user{
background:#1F6FEB;
padding:12px;
border-radius:15px;
color:white;
margin-bottom:10px;
}

.bot{
background:#262730;
padding:15px;
border-radius:15px;
margin-bottom:15px;
}

.metric{
background:#1E1E1E;
padding:15px;
border-radius:10px;
text-align:center;
}

</style>
""",unsafe_allow_html=True)

# ----------------------------
# SIDEBAR
# ----------------------------

with st.sidebar:

    st.title("📚 DSA Knowledge Base")

    st.success("Knowledge Base Loaded")

    st.metric("Concepts",len(engine.concepts))

    st.metric("Categories",7)

    st.metric("Examples",336)

    st.divider()

    st.subheader("Example Questions")

    st.info("""
• What is AVL Tree?

• Applications of Queue

• Advantages of Stack

• Complexity of Binary Heap

• Explain Dynamic Array
""")

# ----------------------------
# TITLE
# ----------------------------

st.title("🤖 DSA AI Assistant")

st.caption("Powered by your custom DSA Knowledge Base")

# ----------------------------
# QUERY
# ----------------------------

query=st.chat_input("Ask anything about DSA...")

if query:

    st.chat_message("user").markdown(query)

    clean=processor.clean_query(query)

    intent=detector.detect(query)

    with st.spinner("Thinking..."):

        results=engine.search_by_name(clean)

    if not results:

        st.chat_message("assistant").error("No matching concept found.")

    else:

        concept=results[0]

        with st.chat_message("assistant"):

            st.subheader(f"📘 {concept['name']}")

            if intent=="definition":

                st.markdown("### 📖 Definition")

                st.write(concept["definition"])

            elif intent=="complexity":

                col1,col2=st.columns(2)

                with col1:
                    st.metric("Time Complexity","See Below")

                    st.code(concept["ops_basic"])

                with col2:
                    st.metric("Space Complexity",concept["space_complexity"])

            elif intent=="applications":

                st.markdown("### 🚀 Applications")

                st.success(concept["applications"])

            elif intent=="advantages":

                st.markdown("### ✅ Advantages")

                st.success(concept["advantages"])

            elif intent=="disadvantages":

                st.markdown("### ❌ Disadvantages")

                st.error(concept["disadvantages"])

            else:

                st.markdown("### 📖 Definition")

                st.write(concept["definition"])

                st.markdown("---")

                col1,col2=st.columns(2)

                with col1:

                    st.markdown("### ✅ Advantages")

                    st.success(concept["advantages"])

                with col2:

                    st.markdown("### ❌ Disadvantages")

                    st.error(concept["disadvantages"])

                st.markdown("---")

                st.markdown("### 🚀 Applications")

                st.info(concept["applications"])

                st.markdown("---")

                st.markdown("### ⚡ Complexity")

                st.code(concept["ops_basic"])

                st.metric("Space Complexity",concept["space_complexity"])