import os
from langsmith import Client, traceable

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]    = "your_langsmith_api_key_here"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"]    = "AI-Lab-Week1-Sidharadha"

client = Client()
print("Connected to LangSmith!")

@traceable(name="GreetingChain")
def greet_user(name: str) -> str:
    return f"Hello {name}! LangSmith setup is working!"

@traceable(name="SimpleQA")
def answer_question(question: str) -> str:
    answers = {
        "what is langsmith": "A tool to trace and monitor AI apps.",
        "what is langchain": "A framework for building AI applications.",
    }
    return answers.get(question.lower(), "Not found!")

if __name__ == "__main__":
    print(greet_user("Sidharadha"))
    print(answer_question("what is langsmith"))
    print(answer_question("what is langchain"))
    print("Done! Check smith.langchain.com for your traces.")