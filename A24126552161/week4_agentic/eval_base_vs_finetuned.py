"""
Week 4 — Agentic Tool Use: Base vs Fine-Tuned Comparison

Loads your Week 3 fine-tuned model (merged with its LoRA adapter) and
compares it against the untouched base model, both wired into the same
LangChain agent + Week 1 tools. Evaluates on a small held-out test set
(from Week 2's test.jsonl) plus a few general tool-use prompts.

Run this on the same GPU pod used for Week 3 training (the fine-tuned
model needs to be loaded for inference here).
"""

import json
import csv
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langgraph.prebuilt import create_react_agent

import sys
sys.path.append("week1_tools")
from tools import ALL_TOOLS, generate_comment


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "week3_sft/final_adapter"
TEST_FILE = "week2_data/test.jsonl"
RESULTS_PATH = "week4_agentic/results/comparison.csv"
N_TEST_EXAMPLES = 10   # keep small — this is a qualitative capstone comparison, not a huge benchmark


def load_model_as_chat(use_adapter: bool):
    """Load either the plain base model, or the base model + LoRA adapter merged,
    and wrap it as a LangChain-compatible chat model."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", torch_dtype=torch.bfloat16
    )

    if use_adapter:
        print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model = model.merge_and_unload()  # merge adapter into base weights for clean inference

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=False,
    )
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    return ChatHuggingFace(llm=llm)


def build_agent(chat_model):
    return create_react_agent(chat_model, ALL_TOOLS)


def run_agent_query(agent, query: str):
    """Run one query through the agent, return final answer + whether a tool was called."""
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = result["messages"]
    tool_called = any(msg.type == "tool" for msg in messages)
    final_answer = messages[-1].content
    return final_answer, tool_called


def load_test_examples(path, n):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            examples.append(json.loads(line))
    return examples


def main():
    Path("week4_agentic/results").mkdir(parents=True, exist_ok=True)

    # General tool-use prompts (test whether the model calls the RIGHT tool at all)
    general_prompts = [
        "What is (45 + 15) * 3?",
        "Save a note that says 'check eval results', then list my notes.",
        "What's the weather in Hyderabad?",
    ]

    test_examples = load_test_examples(TEST_FILE, N_TEST_EXAMPLES)

    rows = []

    for model_label, use_adapter in [("base", False), ("fine_tuned", True)]:
        print(f"\n=== Evaluating {model_label} model ===")
        chat_model = load_model_as_chat(use_adapter)
        agent = build_agent(chat_model)

        # 1. General tool-selection accuracy
        for prompt in general_prompts:
            answer, tool_called = run_agent_query(agent, prompt)
            rows.append({
                "model": model_label,
                "task_type": "general_tool_use",
                "prompt": prompt,
                "answer": answer,
                "tool_called": tool_called,
                "expected_output": "",
            })
            print(f"[{model_label}] '{prompt}' -> tool_called={tool_called}")

        # 2. Code-comment generation via the generate_comment tool, on held-out test code
        for ex in test_examples:
            prompt = f"Generate a comment for this code:\n{ex['input']}"
            answer, tool_called = run_agent_query(agent, prompt)
            rows.append({
                "model": model_label,
                "task_type": "comment_generation",
                "prompt": ex["input"][:200],  # truncate for readability in the CSV
                "answer": answer,
                "tool_called": tool_called,
                "expected_output": ex["output"],
            })

        # free GPU memory before loading the next model
        del chat_model, agent
        torch.cuda.empty_cache()

    # save all results to CSV for manual comparison
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {RESULTS_PATH}")

    # quick summary: tool-call rate per model
    for model_label in ["base", "fine_tuned"]:
        model_rows = [r for r in rows if r["model"] == model_label]
        tool_rate = sum(r["tool_called"] for r in model_rows) / len(model_rows)
        print(f"{model_label}: tool-called on {tool_rate:.0%} of prompts")


if __name__ == "__main__":
    main()