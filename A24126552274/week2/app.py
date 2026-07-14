import streamlit as st

from search import DSASearchEngine
from queryprocessor import QueryProcessor
from intent_detector import IntentDetector

engine = DSASearchEngine("data/dsa.json")
processor = QueryProcessor()
detector = IntentDetector()

st.set_page_config(
    page_title="DSA Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 DSA Chatbot")
st.write("Ask questions about Data Structures and Algorithms.")

query = st.text_input(
    "Ask a question",
    placeholder="Example: What is AVL Tree?"
)

if st.button("Search"):

    if query.strip() == "":
        st.warning("Please enter a question.")

    else:

        clean_query = processor.clean_query(query)
        intent = detector.detect(query)

        results = engine.search_by_name(clean_query)

        if not results:
            st.error("No concept found.")
        else:

            concept = results[0]

            st.success(f"Concept Found: {concept['name']}")

            if intent == "definition":

                st.subheader("Definition")
                st.write(concept["definition"])

            elif intent == "complexity":

                st.subheader("Time Complexity")
                st.code(concept["ops_basic"])

                st.subheader("Space Complexity")
                st.write(concept["space_complexity"])

            elif intent == "applications":

                st.subheader("Applications")
                st.write(concept["applications"])

            elif intent == "advantages":

                st.subheader("Advantages")
                st.write(concept["advantages"])

            elif intent == "disadvantages":

                st.subheader("Disadvantages")
                st.write(concept["disadvantages"])

            else:

                st.subheader("Definition")
                st.write(concept["definition"])

                st.subheader("Applications")
                st.write(concept["applications"])

                st.subheader("Advantages")
                st.write(concept["advantages"])

                st.subheader("Disadvantages")
                st.write(concept["disadvantages"])