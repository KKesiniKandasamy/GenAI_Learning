import os, glob
import numpy as np
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

class RAGAgent:
    def __init__(self, docs_path='docs', top_k=3):
        self.docs_path = docs_path
        self.top_k = top_k
        self._load_docs()

    def _load_docs(self):
        self.docs = []
        for p in sorted(glob.glob(os.path.join(self.docs_path,'*.txt'))):
            with open(p,'r',encoding='utf8') as f:
                self.docs.append((os.path.basename(p), f.read()))
        self._build_index()

    def _build_index(self):
        texts = [t for (_,t) in self.docs]
        if TfidfVectorizer:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            try:
                self.tfidf = self.vectorizer.fit_transform(texts)
            except Exception:
                self.tfidf = None
        else:
            self.vectorizer = None
            self.tfidf = None

    def retrieve(self, query, k=None):
        k = k or self.top_k
        texts = [t for (_,t) in self.docs]
        if self.tfidf is not None:
            qv = self.vectorizer.transform([query])
            sims = cosine_similarity(qv, self.tfidf)[0]
            idxs = list((-sims).argsort()[:k])
            return [self.docs[i] for i in idxs]
        # fallback simple substring score
        scores = []
        q = query.lower()
        for i,(name,txt) in enumerate(self.docs):
            scores.append((i, txt.lower().count(q)))
        scores.sort(key=lambda x: -x[1])
        return [self.docs[i] for i,_ in scores[:k]]

    def answer(self, query, context=None):
        # retrieve top docs and produce a simple synthesized answer
        hits = self.retrieve(query)
        if not hits:
            return "I don't know — no relevant documents found."
        # naive synthesis: return the top snippet from top doc
        out = []
        for name,txt in hits:
            snippet = txt.strip().replace('\n',' ')[:500]
            out.append(f"From {name}: {snippet}...")
        answer = "\n".join(out)
        answer += "\n\n(Answer synthesized from documents.)"
        return answer
