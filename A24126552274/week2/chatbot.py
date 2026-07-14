from search import DSASearchEngine
from queryprocessor import QueryProcessor
from intent_detector import IntentDetector

engine = DSASearchEngine("data/dsa.json")
processor = QueryProcessor()
detector = IntentDetector()

print("\nDSA Search Engine")
print("-" * 30)

while True:

    query = input("\nAsk: ")

    if query.lower() == "exit":
        break

    clean_query = processor.clean_query(query)

    results = engine.search_by_name(clean_query)

    if not results:
        print("No concept found.")
        continue

    concept = results[0]

    intent = detector.detect(query)

    print(f"\nName: {concept['name']}")

    if intent == "definition":
        print("\nDefinition:")
        print(concept["definition"])

    elif intent == "complexity":
        print("\nTime Complexity:")
        print(concept["ops_basic"])
        print("\nSpace Complexity:")
        print(concept["space_complexity"])

    elif intent == "applications":
        print("\nApplications:")
        print(concept["applications"])

    elif intent == "advantages":
        print("\nAdvantages:")
        print(concept["advantages"])

    elif intent == "disadvantages":
        print("\nDisadvantages:")
        print(concept["disadvantages"])

    else:
        print("\nDefinition:")
        print(concept["definition"])

        print("\nApplications:")
        print(concept["applications"])

        print("\nAdvantages:")
        print(concept["advantages"])

        print("\nDisadvantages:")
        print(concept["disadvantages"])