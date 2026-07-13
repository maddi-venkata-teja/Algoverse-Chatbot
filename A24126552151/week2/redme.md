# Algo-Verse Prompt Testing

## Overview

This repository contains the prompt testing work carried out for the **Algo-Verse** and **Algo-Verse V2** applications. The objective of this project is to evaluate the AI model's ability to understand, explain, and generate accurate responses for various Data Structures and Algorithms (DSA) concepts.

The testing focuses on response accuracy, formatting consistency, correctness of time complexities, handling of advanced algorithmic concepts, and structured JSON generation.

---

## Repository Structure

```
algo-verse-testing/
│
├── README.md
├── full_dataset.json
├── concepts_dataset.json
│
└── test_reports/
    ├── v1_test_log.md
    └── v2_test_log.md
```

---

## Project Objectives

- Test AI-generated explanations for DSA concepts.
- Validate correctness of definitions and properties.
- Verify algorithm time complexities.
- Compare the performance of Algo-Verse V1 and Algo-Verse V2.
- Identify prompt failures and formatting inconsistencies.
- Document bugs and improvement areas.

---

## Dataset Used

### 1. full_dataset.json

A comprehensive prompt dataset containing **42,991** instruction-response pairs related to Data Structures and Algorithms.

Each record includes:

- Instruction
- Input
- Output
- Concept
- Category
- Difficulty
- Query Type

### 2. concepts_dataset.json

A structured reference dataset containing core DSA concepts with:

- Name
- Category
- Definition
- Properties
- Operations
- Time Complexities

This dataset serves as the ground truth for validating AI-generated responses.

---

## Testing Methodology

### Level 1 – Core Accuracy Testing

Objective:
Verify whether the AI provides correct definitions, properties, and operation complexities.

Example Prompt:

```
What are the properties and time complexities of a Hash Table?
```

Validation Criteria:

- Correct definition
- Correct properties
- Accurate complexity analysis

---

### Level 2 – Boundary Testing

Objective:

Evaluate the AI's performance on advanced and non-linear concepts.

Tested Topics:

- Trees
- Binary Search Trees
- Heap
- Trie
- Graph
- Dynamic Programming
- Backtracking
- Greedy Algorithms
- Disjoint Set

Validation Criteria:

- Logical correctness
- Response completeness
- Proper formatting
- Error handling

---

### Level 3 – Schema Validation

Objective:

Check whether the AI can generate responses in strict JSON format.

Example Prompt:

```
Generate a JSON entry for Segment Tree using the provided schema.
```

Validation Criteria:

- Valid JSON
- No conversational text
- Schema consistency

---

## Testing Parameters

The following aspects were evaluated during testing:

- Definition Accuracy
- Concept Understanding
- Time Complexity Correctness
- Algorithm Explanation
- JSON Formatting
- Code Generation
- Response Consistency
- Prompt Robustness
- Advanced Concept Handling

---

## Results Summary

### Algo-Verse V1

- Strong performance on basic DSA concepts
- Minor formatting inconsistencies
- Limited explanations for Dynamic Programming and Trie
- JSON responses occasionally contained extra conversational text

Overall Pass Rate:

**80%**

---

### Algo-Verse V2

- Improved response consistency
- Better handling of Graphs and Trees
- Improved JSON formatting
- More accurate algorithm explanations
- Minor improvements still needed for Dynamic Programming

Overall Pass Rate:

**94%**

---

## Bugs Identified

Some common issues observed during testing include:

- Inconsistent JSON formatting
- Missing complexity analysis in certain responses
- Partial explanations for advanced concepts
- Formatting differences across long outputs

Detailed reports are available in:

- `test_reports/v1_test_log.md`
- `test_reports/v2_test_log.md`

---

## Technologies Used

- JSON
- Markdown
- Streamlit Applications
- Git & GitHub
- Manual Prompt Testing

---

## Conclusion

The prompt testing process demonstrates that Algo-Verse V2 provides more accurate, consistent, and structured responses than V1. The testing dataset and reports help identify areas requiring improvement, particularly for advanced algorithmic paradigms and strict output formatting.

---

## Author

Prompt Testing and Documentation completed as part of the Algo-Verse AI Testing Project.