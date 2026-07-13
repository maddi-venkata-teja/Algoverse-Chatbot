# LangSmith Setup - Week 1 Task for S. Siddaradha (A24...177)
# LangSmith is a platform by LangChain for tracing, monitoring, and debugging LLM applications

# ============================================================
# STEP 1: Install required packages
# ============================================================
# pip install langsmith langchain langchain-core openai

# ============================================================
# STEP 2: Configure Environment Variables
# ============================================================
import os

# Set your LangSmith API Key (get from https://smith.langchain.com)
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"]    = "AI-Lab-Week1-Sidharadha"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_api_key_here"  # Replace with your key
os.environ["LANGCHAIN_PROJECT"] = "AI-Lab-Week1-SiddaradhaSetup"  # Your project name

# ============================================================
# STEP 3: Initialize LangSmith Client
# ============================================================
from langsmith import Client

client = Client()
print("✅ LangSmith Client initialized successfully!")

# ============================================================
# STEP 4: Create a Simple Traced Chain using @traceable decorator
# ============================================================
from langsmith import traceable

@traceable(name="SimpleGreetingChain")
def greet_user(name: str) -> str:
    """A simple traced function - every call is logged in LangSmith."""
    response = f"Hello, {name}! Welcome to the AI Lab. Your LangSmith setup is working!"
    return response

@traceable(name="QuestionAnswerChain")
def answer_question(question: str) -> str:
    """Simulates a QA chain - traced in LangSmith dashboard."""
    answers = {
        "what is langsmith": "LangSmith is a platform for tracing and monitoring LLM applications.",
        "what is langchain": "LangChain is a framework for building LLM-powered applications.",
        "what is llm": "LLM stands for Large Language Model, like GPT or Llama.",
    }
    key = question.lower().strip("?")
    return answers.get(key, f"I don't have an answer for: '{question}'. But it was traced!")

# ============================================================
# STEP 5: Run the traced functions
# ============================================================
if __name__ == "__main__":
    print("\n--- Running Traced Functions ---")
    
    # Test 1: Greeting
    result1 = greet_user("Siddaradha")
    print(f"Greeting Result: {result1}")
    
    # Test 2: QA Chain
    result2 = answer_question("What is LangSmith?")
    print(f"QA Result: {result2}")
    
    result3 = answer_question("What is LangChain?")
    print(f"QA Result: {result3}")
    
    # Test 3: List existing projects
    print("\n--- LangSmith Projects ---")
    try:
        projects = list(client.list_projects())
        for project in projects:
            print(f"  📁 Project: {project.name}")
    except Exception as e:
        print(f"  Note: {e}")
    
    print("\n✅ All traces sent to LangSmith Dashboard!")
    print("🔗 View at: https://smith.langchain.com")
    print("📊 Project: AI-Lab-Week1-SiddaradhaSetup")