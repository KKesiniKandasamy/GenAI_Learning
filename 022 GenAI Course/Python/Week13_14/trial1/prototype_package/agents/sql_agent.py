import sqlite3, os
class SQLAgent:
    def __init__(self, db_path='data/sample.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        if not os.path.exists(db_path):
            self._create_sample_db()

    def _create_sample_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL, category TEXT)')
        items = [
            ('Blue Notebook', 4.99, 'stationery'),
            ('Mechanical Keyboard', 129.99, 'electronics'),
            ('Ceramic Mug', 12.50, 'kitchen'),
            ('Noise Cancelling Headphones', 199.00, 'electronics'),
            ('Standing Desk', 399.00, 'furniture'),
        ]
        c.executemany('INSERT INTO products (name,price,category) VALUES (?,?,?)', items)
        conn.commit()
        conn.close()

    def execute(self, query):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute(query)
            if query.strip().lower().startswith('select'):
                rows = c.fetchall()
                # pretty format
                out = []
                for r in rows:
                    out.append(str(r))
                return '\n'.join(out) if out else '(no rows)'
            else:
                conn.commit()
                return 'OK'
        finally:
            conn.close()
