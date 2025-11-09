# Simple content-based recommender that uses RAG documents for item retrieval
class RecommenderAgent:
    def __init__(self, rag_agent):
        self.rag = rag_agent

    def recommend(self, topic, k=5):
        hits = self.rag.retrieve(topic, k=k)
        # return the doc names as recommendations; in a real system map docs to items
        return [name for name,_ in hits]
