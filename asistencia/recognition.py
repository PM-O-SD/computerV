"""DeepFace se importa cuando se necesita el modelo."""
import threading
import os
from pathlib import Path

MODEL = 'Facenet512'
INFERENCE_LOCK = threading.Lock()


class FaceEngine:
    def __init__(self, backend=None):
        self.backend = backend

    def api(self):
        if self.backend is None:
            os.environ.setdefault('DEEPFACE_HOME', str(Path(__file__).resolve().parent.parent / 'data' / 'asistencia'))
            from deepface import DeepFace
            self.backend = DeepFace
        return self.backend

    def represent(self, image):
        faces = self.api().represent(
            img_path=image, model_name=MODEL, detector_backend='opencv',
            enforce_detection=True, align=True,
        )
        if len(faces) != 1:
            raise ValueError('Usa una imagen con exactamente una persona.')
        return [float(value) for value in faces[0]['embedding']]

    def matches(self, embedding, people):
        matches = []
        for person in people:
            result = self.api().verify(
                img1_path=embedding, img2_path=person['embedding'],
                model_name=MODEL, distance_metric='cosine', silent=True,
            )
            if result['verified']:
                matches.append(person)
        return matches

    def identify(self, embedding, people):
        matches = self.matches(embedding, people)
        if len(matches) > 1:
            return 'ambiguous', None
        if not matches:
            return 'unknown', None
        return 'recognized', matches[0]
