import collections, json, os
class Memory:
    def __init__(self, size=8, path='state/memory.json'):
        self.size = size
        self.path = path
        self.buffer = collections.deque(maxlen=size*2)  # store alternating user/assistant
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._load()

    def add_user(self, text):
        self.buffer.append({'role':'user','text':text})
        self._save()

    def add_assistant(self, text):
        self.buffer.append({'role':'assistant','text':text})
        self._save()

    def get_context(self):
        # return the last N turns as a single string
        items = list(self.buffer)[-self.size*2:]
        return '\n'.join([f"{it['role']}: {it['text']}" for it in items])

    def _save(self):
        try:
            with open(self.path,'w',encoding='utf8') as f:
                json.dump(list(self.buffer), f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _load(self):
        try:
            with open(self.path,'r',encoding='utf8') as f:
                arr = json.load(f)
                for it in arr[-self.buffer.maxlen:]:
                    self.buffer.append(it)
        except Exception:
            return
