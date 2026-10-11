"""Persistencia local sin guardar fotografías."""
import csv
import io
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS people (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    embedding TEXT NOT NULL, consent_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attendance (
                    person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    day TEXT NOT NULL, timestamp TEXT NOT NULL,
                    PRIMARY KEY (person_id, day)
                );
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys = ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def people(self, embeddings=False):
        with self.connect() as db:
            rows = db.execute('SELECT * FROM people ORDER BY name, id').fetchall()
        result = []
        for row in rows:
            person = {'id': row['id'], 'name': row['name']}
            if embeddings:
                person['embedding'] = json.loads(row['embedding'])
            result.append(person)
        return result

    def register(self, person_id, name, embedding, now):
        with self.connect() as db:
            db.execute('INSERT INTO people VALUES (?, ?, ?, ?)',
                       (person_id, name, json.dumps(embedding), now.isoformat()))

    def mark(self, person_id, now):
        with self.connect() as db:
            cursor = db.execute('INSERT OR IGNORE INTO attendance VALUES (?, ?, ?)',
                                (person_id, now.date().isoformat(), now.isoformat()))
            return cursor.rowcount == 1

    def attendance(self, day):
        with self.connect() as db:
            return [dict(row) for row in db.execute('''
                SELECT p.id, p.name, a.day, a.timestamp FROM attendance a
                JOIN people p ON p.id = a.person_id WHERE a.day = ?
                ORDER BY a.timestamp, p.id
            ''', (day,))]

    def delete(self, person_id):
        with self.connect() as db:
            return db.execute('DELETE FROM people WHERE id = ?', (person_id,)).rowcount > 0

    def export(self, day):
        output = io.StringIO(newline='')
        writer = csv.writer(output)
        writer.writerow(['ID', 'Nombre', 'Fecha', 'Hora de registro'])
        for row in self.attendance(day):
            values = [row['id'], row['name'], row['day'], row['timestamp']]
            # Evitar que nombres se interpreten como fórmulas al abrir en Excel.
            writer.writerow(["'" + value if value.lstrip().startswith(('=', '+', '-', '@'))
                             else value for value in values])
        return '\ufeff' + output.getvalue()
