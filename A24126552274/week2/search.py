import json


class DSASearchEngine:
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.concepts = self.data["concepts"]

    # -----------------------------
    # Search by Concept Name
    # -----------------------------
    def search_by_name(self, query):
        query = query.lower().strip()

        results = []

        for concept in self.concepts:

            name = concept["name"].lower()

        # exact match
            if query == name:
                results.append(concept)
                continue

        # concept name inside query
            if name in query:
                results.append(concept)
                continue

        # every word of concept appears somewhere
            words = name.split()

            if all(word in query for word in words):
                results.append(concept)

        return results

    # -----------------------------
    # Search by Category
    # -----------------------------
    def search_by_category(self, category):
        results = []

        for concept in self.concepts:
            if concept["category"].lower() == category.lower():
                results.append(concept)

        return results

    # -----------------------------
    # Search by Keyword
    # -----------------------------
    def search_by_keyword(self, keyword):
        keyword = keyword.lower()
        results = []

        for concept in self.concepts:

            for value in concept.values():

                if isinstance(value, str):

                    if keyword in value.lower():
                        results.append(concept)
                        break

        return results

    # -----------------------------
    # Filter by Difficulty
    # -----------------------------
    def filter_by_difficulty(self, difficulty):
        difficulty = difficulty.lower()
        results = []

        for concept in self.concepts:

            if difficulty == "beginner":
                if "beginner_use_case" in concept:
                    results.append(concept)

            elif difficulty == "intermediate":
                if "intermediate_reason" in concept:
                    results.append(concept)

            elif difficulty == "advanced":
                if "advanced_scenario" in concept:
                    results.append(concept)

        return results

    # -----------------------------
    # Sort Alphabetically
    # -----------------------------
    def sort_alphabetically(self):
        return sorted(self.concepts, key=lambda x: x["name"])

    # -----------------------------
    # Get All Concepts
    # -----------------------------
    def get_all(self):
        return self.concepts