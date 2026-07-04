"""
week3_training_logs/topic_rules.py
──────────────────────────────────────
Week 3 — Metadata Tagging
Contribution by: Sushanth (A24126552152)

Keyword-based topic/category/difficulty classifier used to tag
chunks before they're (re-)embedded and stored. Lightweight by
design — swap in a trained classifier later without changing the
public infer_metadata() interface.
"""

TOPIC_KEYWORDS = {
    "Stack": ["stack", "lifo", "push", "pop"],
    "Queue": ["queue", "fifo", "enqueue", "dequeue"],
    "Linked List": ["linked list", "node", "pointer", "next"],
    "Array": ["array", "index", "contiguous"],
    "Binary Tree": ["tree", "binary tree", "root", "leaf", "subtree"],
    "Graph": ["graph", "vertex", "edge", "adjacency", "dfs", "bfs"],
    "Sorting": ["sort", "bubble sort", "merge sort", "quick sort"],
    "Searching": ["search", "binary search", "linear search"],
}

CATEGORY_MAP = {
    "Stack": "Linear Data Structure",
    "Queue": "Linear Data Structure",
    "Linked List": "Linear Data Structure",
    "Array": "Linear Data Structure",
    "Binary Tree": "Non-Linear Data Structure",
    "Graph": "Non-Linear Data Structure",
    "Sorting": "Algorithm",
    "Searching": "Algorithm",
}

DIFFICULTY_KEYWORDS = {
    "Advanced": ["amortized", "np-hard", "dynamic programming", "red-black", "avl"],
    "Intermediate": ["recursion", "complexity", "balanced", "traversal"],
}


def infer_topic(text: str) -> str:
    lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return topic
    return "General"


def infer_category(topic: str) -> str:
    return CATEGORY_MAP.get(topic, "General")


def infer_difficulty(text: str) -> str:
    lower = text.lower()
    for level, keywords in DIFFICULTY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return level
    return "Beginner"


def infer_metadata(text: str, source: str, chapter: str = "") -> dict:
    topic = infer_topic(text)
    return {
        "topic": topic,
        "category": infer_category(topic),
        "difficulty": infer_difficulty(text),
        "source": source,
        "chapter": chapter or topic,
        "source_type": "text",
    }
