class IntentDetector:

    def detect(self, query):

        query = query.lower()

        if "definition" in query or "what is" in query or "explain" in query:
            return "definition"

        elif "complexity" in query or "time complexity" in query:
            return "complexity"

        elif "application" in query or "use case" in query or "used for" in query:
            return "applications"

        elif "advantage" in query or "benefit" in query:
            return "advantages"

        elif "disadvantage" in query or "limitation" in query:
            return "disadvantages"

        elif "compare" in query or "difference" in query:
            return "comparison"

        return "all"