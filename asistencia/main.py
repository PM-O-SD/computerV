import logging
import os
import re
import sqlite3
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .recognition import FaceEngine, INFERENCE_LOCK
from .storage import Store

BASE = Path(__file__).resolve().parent
ZONE = ZoneInfo(os.environ.get('ATTENDANCE_TIMEZONE', 'America/Mexico_City'))
store = Store(BASE.parent / 'data' / 'asistencia' / 'registros.sqlite3')
engine = FaceEngine()
app = FastAPI(title='Asistencia facial local')
app.mount('/static', StaticFiles(directory=BASE / 'static'), name='static')


def now():
    return datetime.now(ZONE)


def selected_day(day):
    if day is None:
        return now().date().isoformat()
    try:
        return date.fromisoformat(day).isoformat()
    except ValueError:
        raise HTTPException(400, 'La fecha debe tener formato AAAA-MM-DD.')


def decode(upload):
    data = upload.file.read(5 * 1024 * 1024 + 1)
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(413, 'La imagen no debe superar 5 MB.')
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, 'Selecciona una imagen JPG o PNG válida.')
    if image.shape[0] * image.shape[1] > 12_000_000:
        raise HTTPException(400, 'Usa una imagen de hasta 12 megapíxeles.')
    return image


def embedding(image):
    try:
        return engine.represent(image)
    except ValueError:
        raise HTTPException(400, 'No se pudo procesar un único rostro. Usa una foto frontal e iluminada.')


@app.exception_handler(Exception)
async def unexpected(request, exc):
    logging.exception('Error al procesar la solicitud', exc_info=exc)
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={'detail': 'No se pudo completar la operación. Revisa la consola y las dependencias del modelo.'})


@app.get('/')
def home():
    return FileResponse(BASE / 'static' / 'index.html')


@app.get('/api/health')
def health():
    return {'status': 'ok', 'model': 'Facenet512', 'timezone': str(ZONE), 'date': now().date().isoformat()}


@app.get('/api/people')
def people():
    return store.people()


@app.post('/api/register', status_code=201)
def register(person_id: str = Form(...), name: str = Form(...),
             consent: bool = Form(...), photo: UploadFile = File(...)):
    person_id, name = person_id.strip(), name.strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', person_id):
        raise HTTPException(400, 'El ID debe contener de 1 a 40 letras, números, guiones o guiones bajos.')
    if not name or len(name) > 100 or any(ord(char) < 32 for char in name):
        raise HTTPException(400, 'Escribe un nombre de 1 a 100 caracteres sin saltos de línea.')
    if not consent:
        raise HTTPException(400, 'El registro requiere autorización de la persona.')
    image = decode(photo)
    with INFERENCE_LOCK:
        current = store.people(embeddings=True)
        if any(person['id'] == person_id for person in current):
            raise HTTPException(409, 'Ese ID ya está registrado.')
        vector = embedding(image)
        if engine.matches(vector, current):
            raise HTTPException(409, 'Este rostro coincide con un registro existente. Revisa la lista antes de continuar.')
        try:
            store.register(person_id, name, vector, now())
        except sqlite3.IntegrityError:
            raise HTTPException(409, 'Ese ID ya está registrado.')
    return {'message': 'Persona registrada.', 'id': person_id, 'name': name}


@app.post('/api/recognize')
def recognize(photo: UploadFile = File(...)):
    image = decode(photo)
    with INFERENCE_LOCK:
        current = store.people(embeddings=True)
        if not current:
            raise HTTPException(400, 'Registra una persona primero.')
        status, person = engine.identify(embedding(image), current)
        if status == 'unknown':
            return {'status': status, 'message': 'Rostro no reconocido. No se registró asistencia.'}
        if status == 'ambiguous':
            return {'status': status, 'message': 'Hay más de una coincidencia. Revisa manualmente; no se registró asistencia.'}
        created = store.mark(person['id'], now())
    return {'status': 'registered' if created else 'already_registered',
            'name': person['name'], 'id': person['id'],
            'message': 'Asistencia registrada.' if created else 'La asistencia de hoy ya estaba registrada.'}


@app.get('/api/attendance')
def attendance(day: str | None = None):
    return store.attendance(selected_day(day))


@app.get('/api/export')
def export(day: str | None = None):
    day = selected_day(day)
    return Response(store.export(day), media_type='text/csv; charset=utf-8',
                    headers={'Content-Disposition': f'attachment; filename="asistencia_{day}.csv"'})


@app.delete('/api/people/{person_id}')
def delete_person(person_id: str):
    with INFERENCE_LOCK:
        if not store.delete(person_id):
            raise HTTPException(404, 'Persona no encontrada.')
    return {'message': 'Registro facial y asistencias asociadas eliminados.'}
