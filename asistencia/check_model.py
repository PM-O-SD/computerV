"""Comprobación opcional del modelo real sin fotografías personales."""
import numpy as np

from asistencia.recognition import FaceEngine


def main():
    engine = FaceEngine()
    api = engine.api()
    api.build_model('Facenet512')
    first = [1.0] + [0.0] * 511
    second = [0.0, 1.0] + [0.0] * 510
    assert api.verify(first, first, model_name='Facenet512', distance_metric='cosine', silent=True)['verified']
    assert not api.verify(first, second, model_name='Facenet512', distance_metric='cosine', silent=True)['verified']
    print('Modelo cargado y verificación de vectores sintéticos correcta.')
    try:
        engine.represent(np.zeros((256, 256, 3), dtype=np.uint8))
    except ValueError:
        print('Imagen sin rostro rechazada correctamente.')
    else:
        raise AssertionError('Se aceptó una imagen sin rostro.')
    print('Esta comprobación no mide la precisión de reconocimiento de personas.')


if __name__ == '__main__':
    main()
