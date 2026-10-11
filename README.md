# Sistema de asistencia facial y MoodMeter

## Sistema de asistencia para la entrega de visión por computadora

La nueva aplicación registra personas autorizadas, reconoce un rostro mediante una fotografía o cámara y registra una asistencia por persona y día. Puedes consultar una fecha y descargar sus asistencias como CSV. La aplicación MoodMeter original se conserva en `app/`.

### Instalación recomendada

Usa Python 3.11 de 64 bits. Desde la carpeta del repositorio, crea un entorno aislado e instala las dependencias específicas de asistencia:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r asistencia/requirements.txt
.\.venv\Scripts\python.exe -m uvicorn asistencia.main:app --host 127.0.0.1 --port 8001
```

En macOS/Linux:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r asistencia/requirements.txt
.venv/bin/python -m uvicorn asistencia.main:app --host 127.0.0.1 --port 8001
```

Abre **http://127.0.0.1:8001**. No necesitas activar el entorno virtual con estos comandos. TensorFlow y DeepFace son dependencias grandes; la instalación y la primera descarga del modelo pueden tardar varios minutos y requieren conexión. No mezcles este entorno con una instalación previa de MoodMeter. La configuración fijada usa TensorFlow; el DeepFace más reciente puede tener otra configuración predeterminada.

El `requirements.txt` raíz incluye las dependencias de ambas aplicaciones, pero para reproducir esta entrega usa las versiones de `asistencia/requirements.txt`.

### Demostración paso a paso

1. Escribe un ID y nombre, selecciona una foto frontal con un solo rostro y confirma la autorización de la persona. Pulsa **Registrar persona**.
2. En **Tomar asistencia**, sube una segunda foto de la misma persona o activa la cámara y pulsa **Capturar y reconocer**.
3. Opcionalmente activa el reconocimiento automático cada cinco segundos. Sólo se realiza una solicitud a la vez.
4. Comprueba la asistencia y vuelve a reconocer a la persona: debe indicar que ya estaba registrada ese día.
5. Descarga el CSV. Un rostro desconocido o una coincidencia ambigua no registra asistencia.
6. Para retirar un registro, pulsa **Eliminar** y confirma; también se eliminan sus asistencias.

La cámara funciona en localhost y requiere permiso del navegador. Si no hay cámara, el flujo con fotografías permite demostrar todo el proyecto.

### Funcionamiento y datos

DeepFace detecta y alinea el rostro con OpenCV, obtiene un embedding con Facenet512 y verifica su similitud frente a los registros mediante distancia coseno y el umbral predeterminado del modelo. Se rechazan imágenes con varios rostros. Los nombres, embeddings y asistencias se guardan en SQLite bajo `data/asistencia/`, excluida de Git. Las fotografías se procesan en memoria y no se guardan. Los embeddings también son datos biométricos; usa únicamente participantes autorizados.

La fecha se calcula en `America/Mexico_City`. Es una demostración local sin autenticación ni prueba de vida: no debe usarse como control de acceso o asistencia oficial sin evaluación adicional. Una fotografía podría generar una coincidencia. El reconocimiento puede fallar con mala iluminación o cambios de apariencia; revisar los registros sigue siendo necesario. No se infieren edad, género ni emociones para tomar asistencia.

### Comprobaciones

```sh
python -m unittest asistencia.test_core asistencia.test_api -v
```

Las pruebas comprueban que solicitudes concurrentes no duplican asistencias, que otro día permite un nuevo registro, que la eliminación retira los datos asociados, que el CSV no interpreta nombres como fórmulas y que coincidencias ambiguas/desconocidas y varias caras se manejan correctamente. Una prueba HTTP recorre registro, reconocimiento, consulta, exportación y eliminación. Las pruebas de decisiones usan un backend simulado e imágenes sintéticas; no acreditan precisión biométrica.

Para validar el modelo real, registra participantes autorizados y prueba fotos diferentes, un rostro no registrado, imágenes sin rostro y con dos personas. Documenta coincidencias correctas e incorrectas antes de afirmar una precisión. No se incluyen fotografías de personas ni resultados biométricos inventados.

Comprobación opcional de la carga del modelo y de rechazo de imágenes sin rostro:

```sh
python -m asistencia.check_model
```

Validación realizada: instalación de las dependencias en Python 3.11.9 sobre Windows, seis pruebas automáticas satisfactorias, carga de Facenet512, verificación de vectores sintéticos y rechazo de una imagen vacía. No se ha validado aún la precisión con personas reales ni la captura desde una cámara física.

### Subir la entrega

Sube `asistencia/`, `requirements.txt`, `.gitignore` y este README. No subas `.venv/` ni `data/asistencia/`. Toma una captura del repositorio actualizado y otra de la aplicación con datos de demostración autorizados.

Fuente de la integración: https://github.com/serengil/deepface/tree/v0.0.93

---

# MoodMeter 📊🎭 — Decodificador de Expresiones Faciales con IA

**MoodMeter** es un playground interactivo y didáctico de visión por computadora diseñado para analizar emociones y microexpresiones faciales en tiempo real. 

Impulsado por **FastAPI** en el backend y la librería de aprendizaje profundo **DeepFace**, el sistema traduce capturas de video en distribuciones de probabilidad emocional usando una **Red Neuronal Convolucional (CNN)** de última generación y despliega un dashboard con estética premium **Glassmorphism en Modo Oscuro** junto con explicaciones conceptuales integradas.

---

## 🧠 ¿Cómo Decodifica la IA las Emociones?

A diferencia de las computadoras, los seres humanos percibimos expresiones faciales de forma instantánea y subconsciente. Para que una máquina replique esto, implementa un pipeline convolucional jerárquico paso a paso:

```
[ Entrada: Webcam o Foto ]
          │
          ▼
[ Detector de Rostro (OpenCV) ] ──> Recorta el rostro y alinea ojos y boca.
          │
          ▼
[ Capas Convolucionales de la CNN ] ──> Extraen rasgos abstractos (comisura de labios, cejas).
          │
          ▼
[ Capa Completamente Conectada ] ──> Genera 7 puntuaciones brutas (logits) de emoción.
          │
          ▼
[ Función de Activación Softmax ] ──> Normaliza los logits en probabilidades (suman 100%).
          │
          ├───────────────────────────┐
          ▼                           ▼
[ Medidores Neón en Vivo ]     [ Curva Dinámica (Mood Timeline) ]
```

### La Matemática Detrás: Función Softmax
La última capa de la red de emociones arroja puntuaciones crudas ($z_i$) para cada emoción básica. Para traducir estas puntuaciones abstractas en porcentajes legibles que sumen exactamente **100%**, la red implementa la función matemática **Softmax**:

$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

* La función aplica un operador exponencial ($e^{z_i}$) a cada puntuación para asegurar que todos los valores sean estrictamente positivos.
* Luego, divide cada valor por la suma de todos los exponenciales ($\sum e^{z_j}$), convirtiendo los números en una distribución de probabilidad coherente.
* La probabilidad más alta indica la **emoción dominante** calculada por la IA.

---

## 📂 Estructura del Repositorio

La arquitectura del proyecto está optimizada para ser ligera, rápida y 100% libre de bases de datos persistentes:

```
computerV/
├── app/
│   ├── main.py                 # Servidor FastAPI con pipeline de inferencia DeepFace
│   └── templates/
│       ├── index.html          # Frontend premium en HTML5 con Canvas y webcam
│       └── style.css           # Estilos de alta gama (neón, vidrio esmerilado y animaciones)
├── requirements.txt            # Dependencias del proyecto de Python
└── start.sh                    # Script bash para instalación y arranque automático
```

---

## 🛠️ Instalación y Arranque Automático

Sigue estos sencillos pasos para instalar y poner en marcha **MoodMeter** en tu máquina local:

### 1. Navegar al Directorio del Proyecto
```bash
cd /Users/leonelmendiola/computerV
```

### 2. Arrancar la Aplicación
El sistema incluye un script automatizado `start.sh` que se encargará de crear el entorno virtual (`venv`), actualizar `pip`, instalar todas las dependencias y levantar el servidor FastAPI:
```bash
./start.sh
```

### 3. Abrir el Navegador
Una vez que el servidor reporte estar listo, abre tu navegador en:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

> 💡 **Nota de Primera Ejecución**: La primera vez que presiones "Escanear Rostro", DeepFace descargará automáticamente los pesos preentrenados del clasificador de emociones de Keras (~20MB). Este proceso se realiza una sola vez de forma interna y tardará solo unos segundos según tu velocidad de internet.

---

## 💻 Características del Dashboard

* **Captura Multidispositivo**: Si cuentas con cámara web activa, la rejilla láser cibernética escaneará tu rostro en vivo. Si no posees cámara web, haz clic en **Archivo** para subir cualquier fotografía local.
* **Mira Láser Flotante**: Al completar el escaneo, verás un recuadro cibernético dashed de color cian dibujado exactamente sobre las coordenadas físicas de tu rostro.
* **Gauges de Distribución**: Muestra barras neón de colores que representan el peso asignado por la red a cada emoción en tiempo real.
* **Mood Timeline (Línea de Tiempo)**: Un gráfico dinámico hecho con HTML5 Canvas que conecta los puntos de tus escaneos secuenciales. La curva cambia de color dinámicamente según la emoción dominante registrada en cada paso, ilustrando tu transición emocional (ej. de enojo a alegría).

---

## ⚡ Tecnologías Utilizadas
* **FastAPI** — API de alta velocidad en Python.
* **DeepFace** — Framework de análisis facial inmersivo.
* **OpenCV** — Localización y procesamiento espacial del rostro.
* **HTML5 / CSS3 / JavaScript (ES6)** — Frontend premium inmersivo sin dependencias pesadas.
