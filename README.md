# Verificación facial en tiempo real para repartidores

Sistema que usa la cámara web para verificar la identidad de una persona comparando su rostro con un registro previo. Es un prototipo desarrollado en el curso **Percepción y Visión por Computadora** de la Universidad Nacional de Trujillo.

## Cómo funciona

1. **Detección:** se detecta el rostro en cada fotograma con Haar Cascades (OpenCV).
2. **Embedding:** una red convolucional (MobileNetV2 con capas densas de 512 y 256 unidades) convierte el rostro, a 224×224, en un vector de 128 dimensiones normalizado con L2.
3. **Comparación:** el vector se compara por similitud coseno con los embeddings registrados. Si el valor supera el umbral (0,6 por defecto), la persona queda verificada.
4. **Resultado:** se dibuja sobre el video el estado de la verificación y el nivel de confianza. El procesamiento de fotogramas corre en un hilo aparte para mantener fluido el video.

## Tecnologías

Python 3, TensorFlow y Keras, OpenCV, NumPy, scikit-learn.

## Archivos

| Archivo | Descripción |
|---|---|
| `0realtime_face_verification.py` | Aplicación de verificación en tiempo real |

## Cómo ejecutarlo

```bash
pip install tensorflow opencv-python numpy scikit-learn
python 0realtime_face_verification.py
```

Con la ventana de video activa, las teclas `+` y `-` ajustan el umbral.

> **Importante:** el modelo entrenado (`embedding_model.h5`) y la base de embeddings (`embeddings_db.pkl` o `embeddings_db.json`) no están incluidos en el repositorio. Deben colocarse en una carpeta `models/<nombre>` y se indica su ruta en el constructor de `RealtimeFaceVerification`.

## Pendiente

- Incluir el script de entrenamiento y de registro de personas.
- Añadir un archivo `requirements.txt`.
- Evaluar la precisión con un conjunto de prueba.

## Autor

Serge López, estudiante de Ingeniería Informática en la Universidad Nacional de Trujillo.
