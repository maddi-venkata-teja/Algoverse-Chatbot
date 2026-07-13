
import json
import random

# Load dataset
with open("dsa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Get the list of examples
examples = data["training_examples"]

# Shuffle the examples
random.seed(42)
random.shuffle(examples)

# Split ratios
n = len(examples)

train_end = int(0.8 * n)
val_end = int(0.9 * n)

train_data = examples[:train_end]
val_data = examples[train_end:val_end]
test_data = examples[val_end:]

# Save train.json
with open("train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=4)

# Save validation.json
with open("validation.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, indent=4)

# Save test.json
with open("test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=4)

print("===================================")
print("Dataset Split Completed Successfully!")
print("===================================")
print(f"Total Examples : {n}")
print(f"Train          : {len(train_data)}")
print(f"Validation     : {len(val_data)}")
print(f"Test           : {len(test_data)}")