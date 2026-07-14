import re

class QueryProcessor:

    def clean_query(self, query):
        query = query.lower()

        # Remove punctuation
        query = re.sub(r"[^\w\s]", "", query)

        # Common words to ignore
        stop_words = {
            "what", "is", "the", "a", "an",
            "tell", "me", "about",
            "explain", "describe",
            "please", "give", "define"
        }

        words = query.split()

        words = [word for word in words if word not in stop_words]

        return " ".join(words)