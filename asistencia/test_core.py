import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from asistencia.storage import Store
from asistencia.recognition import FaceEngine


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / 'test.sqlite3')
        self.now = datetime(2026, 10, 11, 10, tzinfo=timezone.utc)
        self.store.register('01', 'Ana', [0.1], self.now)

    def test_one_record_per_day_even_concurrently(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: self.store.mark('01', self.now), range(8)))
        self.assertEqual(sum(results), 1)
        self.assertEqual(len(self.store.attendance('2026-10-11')), 1)
        self.assertTrue(self.store.mark('01', self.now + timedelta(days=1)))

    def test_delete_removes_attendance_and_embedding(self):
        self.store.mark('01', self.now)
        self.assertTrue(self.store.delete('01'))
        self.assertEqual(self.store.people(embeddings=True), [])
        self.assertEqual(self.store.attendance('2026-10-11'), [])

    def test_csv_formula_is_escaped(self):
        self.store.register('02', '=1+1', [0.2], self.now)
        self.store.mark('02', self.now)
        self.assertIn("'=1+1", self.store.export('2026-10-11'))
        self.assertNotIn('embedding', self.store.export('2026-10-11'))


class Backend:
    def represent(self, **kwargs):
        return [{'embedding': [1.0]}]

    def verify(self, **kwargs):
        return {'verified': kwargs['img1_path'] == kwargs['img2_path']}


class RecognitionTests(unittest.TestCase):
    def test_unknown_recognized_and_ambiguous(self):
        engine = FaceEngine(Backend())
        ana = {'id': '01', 'name': 'Ana', 'embedding': [1.0]}
        self.assertEqual(engine.identify([2.0], [ana]), ('unknown', None))
        self.assertEqual(engine.identify([1.0], [ana]), ('recognized', ana))
        self.assertEqual(engine.identify([1.0], [ana, {**ana, 'id': '02'}]), ('ambiguous', None))

    def test_multiple_faces_are_rejected(self):
        backend = Backend()
        backend.represent = lambda **kwargs: [{'embedding': [1]}, {'embedding': [2]}]
        with self.assertRaises(ValueError):
            FaceEngine(backend).represent(None)


if __name__ == '__main__':
    unittest.main()
