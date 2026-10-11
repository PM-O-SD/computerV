"""Pruebas HTTP con imágenes sintéticas y reconocimiento simulado."""
import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import uvicorn

from asistencia import main
from asistencia.storage import Store


class FakeEngine:
    def represent(self, image):
        return [float(image.mean())]

    def matches(self, embedding, people):
        return [p for p in people if p['embedding'] == embedding]

    def identify(self, embedding, people):
        matches = self.matches(embedding, people)
        return ('recognized', matches[0]) if len(matches) == 1 else ('unknown', None)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.previous_store, cls.previous_engine = main.store, main.engine
        main.store = Store(Path(cls.tmp.name) / 'test.sqlite3')
        main.engine = FakeEngine()
        cls.socket = socket.socket()
        cls.socket.bind(('127.0.0.1', 0))
        cls.base = 'http://127.0.0.1:' + str(cls.socket.getsockname()[1])
        cls.server = uvicorn.Server(uvicorn.Config(main.app, log_level='error'))
        cls.thread = threading.Thread(target=cls.server.run, kwargs={'sockets': [cls.socket]}, daemon=True)
        cls.thread.start()
        deadline = time.monotonic() + 10
        while not cls.server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        if not cls.server.started:
            raise RuntimeError('El servidor no arrancó.')

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=10)
        cls.socket.close()
        main.store, main.engine = cls.previous_store, cls.previous_engine
        cls.tmp.cleanup()

    def request(self, path, method='GET', fields=None, pixel=200, raw=False):
        data = None
        headers = {}
        if fields is not None:
            boundary = 'attendance-test-boundary'
            chunks = []
            for name, value in fields.items():
                chunks.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode())
            image = cv2.imencode('.png', np.full((32, 32, 3), pixel, np.uint8))[1].tobytes()
            chunks.append((f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; filename="test.png"\r\nContent-Type: image/png\r\n\r\n').encode() + image + b'\r\n')
            chunks.append(f'--{boundary}--\r\n'.encode())
            data = b''.join(chunks)
            headers['Content-Type'] = 'multipart/form-data; boundary=' + boundary
        request = Request(self.base + path, data=data, headers=headers, method=method)
        try:
            response = urlopen(request, timeout=10)
        except HTTPError as error:
            response = error
        with response:
            body = response.read().decode('utf-8-sig')
            return response.status, body if raw else json.loads(body)

    def test_registration_recognition_export_and_deletion(self):
        self.assertEqual(self.request('/api/health')[0], 200)
        self.assertIn('Asistencia facial', self.request('/', raw=True)[1])
        fields = {'person_id': 'test_01', 'name': 'Persona de prueba', 'consent': 'false'}
        self.assertEqual(self.request('/api/register', 'POST', fields)[0], 400)
        fields['consent'] = 'true'
        self.assertEqual(self.request('/api/register', 'POST', fields)[0], 201)
        self.assertEqual(self.request('/api/register', 'POST', fields)[0], 409)
        self.assertEqual(self.request('/api/recognize', 'POST', {}, pixel=150)[1]['status'], 'unknown')
        self.assertEqual(self.request('/api/attendance')[1], [])
        self.assertEqual(self.request('/api/recognize', 'POST', {})[1]['status'], 'registered')
        self.assertEqual(self.request('/api/recognize', 'POST', {})[1]['status'], 'already_registered')
        self.assertEqual(len(self.request('/api/attendance')[1]), 1)
        self.assertIn('Persona de prueba', self.request('/api/export', raw=True)[1])
        self.assertNotIn('embedding', self.request('/api/people', raw=True)[1])
        self.assertEqual(self.request('/api/attendance?day=incorrecta')[0], 400)
        self.assertEqual(self.request('/api/people/test_01', 'DELETE')[0], 200)
        self.assertEqual(self.request('/api/attendance')[1], [])
        self.assertEqual(self.request('/api/people')[1], [])


if __name__ == '__main__':
    unittest.main()
