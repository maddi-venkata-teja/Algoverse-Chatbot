import json

sample_data = [
    {
        "algorithm": "Binary Search",
        "difficulty": "Easy",
        "input": [2, 5, 8, 10, 15],
        "target": 10,
        "output": 3,
        "explanation": "Target found at index 3."
    },
    {
        "algorithm": "Binary Search",
        "difficulty": "Medium",
        "input": [1, 4, 7, 9, 12, 18],
        "target": 5,
        "output": -1,
        "explanation": "Target not found."
    }
]

with open("../datasets/binarysearch.json", "w") as file:
    json.dump(sample_data, file, indent=4)

print("Dataset created successfully!")