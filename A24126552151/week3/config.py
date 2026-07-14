"""
config.py
----------

Central configuration for the synthetic dataset generator.
Change only this file if you want to generate more or fewer examples.
"""

# ============================================================
# Output
# ============================================================

OUTPUT_DIR = "outputs"

FULL_DATASET_FILE = "outputs/full_dataset.json"
TRAIN_FILE = "outputs/train.jsonl"
VALIDATION_FILE = "outputs/validation.jsonl"
TEST_FILE = "outputs/test.jsonl"

STATISTICS_FILE = "outputs/statistics.json"


# ============================================================
# Dataset Split
# ============================================================

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10


# ============================================================
# Random Seed
# ============================================================

RANDOM_SEED = 42


# ============================================================
# Dataset Size
# ============================================================

# Number of instruction variants to generate
QUESTION_VARIANTS = 16

# Number of response styles
RESPONSE_VARIANTS = 8


# ============================================================
# Duplicate Handling
# ============================================================

REMOVE_DUPLICATES = True


# ============================================================
# Shuffle
# ============================================================

SHUFFLE_DATASET = True


# ============================================================
# Validation
# ============================================================

VALIDATE_BEFORE_SAVE = True


# ============================================================
# Export Options
# ============================================================

EXPORT_JSON = True
EXPORT_JSONL = True


# ============================================================
# Pretty Print JSON
# ============================================================

JSON_INDENT = 2


# ============================================================
# Balance Checks
# ============================================================

CHECK_CATEGORY_BALANCE = True
CHECK_QUERY_BALANCE = True
CHECK_DIFFICULTY_BALANCE = True
CHECK_CONCEPT_BALANCE = True


# ============================================================
# Future Expansion
# ============================================================

ENABLE_INTERVIEW_STYLE = True
ENABLE_REAL_WORLD_STYLE = True
ENABLE_SCENARIO_STYLE = True
ENABLE_COMPARISON_STYLE = True
ENABLE_DEBUG_STYLE = False
ENABLE_CODE_STYLE = False
ENABLE_MULTI_TURN = False