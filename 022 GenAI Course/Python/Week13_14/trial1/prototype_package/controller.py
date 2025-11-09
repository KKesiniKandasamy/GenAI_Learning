import os, re
from agents.memory import Memory
from agents.rag_agent import RAGAgent
from agents.weather_agent import WeatherAgent
from agents.image_agent import ImageAgent
from agents.sql_agent import SQLAgent
from agents.recommender import RecommenderAgent

class Controller:
    def __init__(self, memory_size=10):
        self.memory = Memory(size=memory_size)
        self.rag = RAGAgent(docs_path='docs')
        self.weather = WeatherAgent()
        self.image = ImageAgent(output_dir='outputs')
        self.sql = SQLAgent(db_path='data/sample.db')
        self.recommender = RecommenderAgent(self.rag)

    def handle(self, user_text: str) -> str:
        # store user turn
        self.memory.add_user(user_text)

        # Simple intent detection (rule-based). In production replace with NLU.
        text = user_text.strip()
        low = text.lower()
        if low.startswith('qa:') or low.startswith('question:') or 'what is' in low:
            query = re.sub(r'^(qa:|question:)', '', text, flags=re.I).strip()
            answer = self.rag.answer(query)
            self.memory.add_assistant(answer)
            return answer
        if low.startswith('generate image:') or low.startswith('image:') or 'generate image' in low:
            prompt = re.sub(r'^(generate image:|image:)', '', text, flags=re.I).strip()
            path = self.image.generate(prompt)
            resp = f"Generated image saved to: {path}"
            self.memory.add_assistant(resp)
            return resp
        if low.startswith('weather') or 'weather' in low:
            # extract location crude
            m = re.search(r'in (.+)$', text, flags=re.I)
            location = m.group(1).strip() if m else 'San Francisco'
            w = self.weather.get_weather(location)
            self.memory.add_assistant(w)
            return w
        if low.startswith('sql:') or low.startswith('query:') or low.strip().lower().startswith('select'):
            q = re.sub(r'^(sql:|query:)', '', text, flags=re.I).strip()
            try:
                res = self.sql.execute(q)
                self.memory.add_assistant(str(res))
                return str(res)
            except Exception as e:
                err = f"SQL error: {e}"
                self.memory.add_assistant(err)
                return err
        if low.startswith('recommend:') or 'recommend' in low:
            topic = re.sub(r'^(recommend:)', '', text, flags=re.I).strip()
            if not topic:
                topic = text
            recs = self.recommender.recommend(topic, k=5)
            out = "Recommendations:\n" + "\n".join([f"- {r}" for r in recs])
            self.memory.add_assistant(out)
            return out

        # fallback: echo with memory-aware context using RAG over conversation + docs
        context = self.memory.get_context()
        # try to answer using RAG with user text + context
        composite = f"{context}\nUser: {text}"
        answer = self.rag.answer(text, context=context)
        self.memory.add_assistant(answer)
        return answer
