#!/usr/bin/env python3
"""
Verificación Facial en Tiempo Real
Sistema que usa la cámara web para verificar personas en tiempo real
Basado en el prototipo de embeddings + similaridad coseno
"""

import cv2
import numpy as np
import tensorflow as tf
import pickle
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import time
from datetime import datetime
import threading
import queue

class RealtimeFaceVerification:
    """
    Sistema de verificación facial en tiempo real
    - Captura video de cámara web
    - Detecta rostros con Haar Cascades
    - Verifica identidad con modelo de embeddings
    - Muestra resultados en tiempo real
    """
    
    def __init__(self, model_path="models/face_embeddings_20250617_183416"):
        self.model_path = Path(model_path)
        self.threshold = 0.6
        self.model = None
        self.embeddings_db = {}
        self.face_cascade = None
        
        # Variables para video
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=10)
        
        # Variables para mostrar resultados
        self.last_result = {"person": "Procesando...", "confidence": 0, "status": ""}
        self.detection_color = (0, 255, 0)  # Verde por defecto
        
        # Configuración de visualización
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        
        # Inicializar sistema
        self.load_system()
        self.load_face_detector()
    
    def normalize_layer(self, x):
        """Función de normalización L2 personalizada"""
        return tf.nn.l2_normalize(x, axis=1)
    
    def rebuild_model_from_weights(self, model_file):
        """Reconstruye el modelo manualmente si hay problemas con capas personalizadas"""
        try:
            print("🔧 Reconstruyendo modelo desde arquitectura conocida...")
            
            # Crear modelo base similar al original
            input_layer = tf.keras.layers.Input(shape=(224, 224, 3), name='face_input')
            
            # Backbone CNN (MobileNetV2 sin clasificador)
            backbone = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet',
                pooling='avg'
            )(input_layer)
            
            # Capas de embedding
            x = tf.keras.layers.Dense(512, activation='relu', name='dense_512')(backbone)
            x = tf.keras.layers.BatchNormalization(name='bn_512')(x)
            x = tf.keras.layers.Dropout(0.3, name='dropout_512')(x)
            
            x = tf.keras.layers.Dense(256, activation='relu', name='dense_256')(x)
            x = tf.keras.layers.BatchNormalization(name='bn_256')(x)
            x = tf.keras.layers.Dropout(0.2, name='dropout_256')(x)
            
            # Capa de embedding final
            embeddings = tf.keras.layers.Dense(128, activation=None, name='embeddings')(x)
            
            # Normalización L2 manual
            embeddings_normalized = tf.keras.layers.Lambda(
                lambda x: tf.nn.l2_normalize(x, axis=1),
                name='l2_normalize'
            )(embeddings)
            
            new_model = tf.keras.Model(inputs=input_layer, outputs=embeddings_normalized)
            
            # Intentar cargar pesos del modelo original
            try:
                print("🔄 Intentando cargar pesos entrenados...")
                new_model.load_weights(str(model_file), by_name=True, skip_mismatch=True)
                print("✅ Pesos entrenados cargados exitosamente")
                return new_model
                        
            except Exception as e:
                print(f"⚠️ Error cargando pesos entrenados: {e}")
                print("🔄 Usando modelo con pesos de ImageNet (solo para demostración)")
                print("⚠️ Los resultados serán aleatorios - necesitas el modelo entrenado")
                return new_model
                
        except Exception as e:
            print(f"❌ Error reconstruyendo modelo: {e}")
            return None
    
    def load_system(self):
        """Carga modelo y base de datos de embeddings"""
        print("🎥 SISTEMA DE VERIFICACIÓN FACIAL EN TIEMPO REAL")
        print("="*55)
        print("🔄 Cargando sistema...")
        
        try:
            # Verificar archivos requeridos
            model_file = self.model_path / "embedding_model.h5"
            embeddings_file = self.model_path / "embeddings_db.pkl"
            json_file = self.model_path / "embeddings_db.json"
            
            if not model_file.exists():
                raise FileNotFoundError(f"Modelo no encontrado: {model_file}")
            
            # Cargar modelo de embeddings
            print(f"📦 Cargando modelo: {model_file}")
            
            custom_objects = {
                'Normalize': tf.keras.utils.normalize,
                'normalize': tf.keras.utils.normalize,
                'l2_normalize': tf.nn.l2_normalize,
                'L2Normalize': tf.keras.layers.Lambda(self.normalize_layer)
            }
            
            try:
                self.model = tf.keras.models.load_model(
                    str(model_file), 
                    custom_objects=custom_objects,
                    compile=False
                )
                print("✅ Modelo cargado con objetos personalizados")
                
            except Exception as e1:
                print(f"⚠️ Intento 1 falló: {str(e1)[:100]}...")
                
                try:
                    with tf.keras.utils.custom_object_scope(custom_objects):
                        self.model = tf.keras.models.load_model(str(model_file), compile=False)
                    print("✅ Modelo cargado con scope personalizado")
                    
                except Exception as e2:
                    print(f"⚠️ Intento 2 falló: {str(e2)[:100]}...")
                    
                    # Estrategia 3: Reconstruir modelo
                    print("🔄 Intentando reconstruir modelo...")
                    self.model = self.rebuild_model_from_weights(model_file)
                    
                    if self.model is None:
                        raise Exception("No se pudo cargar el modelo con ningún método")
                    else:
                        print("✅ Modelo reconstruido exitosamente")
            
            # Cargar base de datos de embeddings
            if embeddings_file.exists():
                print(f"📊 Cargando embeddings desde pickle: {embeddings_file}")
                with open(embeddings_file, 'rb') as f:
                    self.embeddings_db = pickle.load(f)
                    
            elif json_file.exists():
                print(f"📊 Cargando embeddings desde JSON: {json_file}")
                with open(json_file, 'r') as f:
                    self.embeddings_db = json.load(f)
                    
            else:
                print("⚠️ Base de datos de embeddings no encontrada")
                self.embeddings_db = {}
            
            # Mostrar información del sistema
            if self.embeddings_db:
                print(f"✅ Base de datos cargada: {len(self.embeddings_db)} personas")
                print("👥 Personas registradas:")
                for i, person in enumerate(self.embeddings_db.keys(), 1):
                    print(f"  {i:2d}. {person}")
            else:
                print("📝 Base de datos vacía - necesitas agregar personas primero")
            
        except Exception as e:
            print(f"❌ Error cargando sistema: {e}")
            raise e
    
    def load_face_detector(self):
        """Carga el detector de rostros Haar Cascade"""
        try:
            # Intentar cargar desde OpenCV
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            if self.face_cascade.empty():
                raise Exception("No se pudo cargar Haar Cascade")
            
            print("✅ Detector de rostros cargado (Haar Cascade)")
            
        except Exception as e:
            print(f"❌ Error cargando detector de rostros: {e}")
            print("💡 Asegúrate de que OpenCV esté instalado correctamente")
            raise e
    
    def preprocess_face(self, face_img):
        """Preprocesa imagen de rostro para el modelo"""
        try:
            # Redimensionar a 224x224
            face_resized = cv2.resize(face_img, (224, 224))
            
            # Convertir a RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Normalizar [0-1]
            face_normalized = face_rgb.astype(np.float32) / 255.0
            
            return face_normalized
            
        except Exception as e:
            print(f"❌ Error preprocesando rostro: {e}")
            return None
    
    def generate_embedding(self, face_img):
        """Genera embedding de 128D para un rostro"""
        try:
            # Agregar dimensión batch
            face_batch = np.expand_dims(face_img, axis=0)
            
            # Generar embedding
            embedding = self.model.predict(face_batch, verbose=0)[0]
            
            # Verificar normalización
            norm = np.linalg.norm(embedding)
            if abs(norm - 1.0) > 0.01:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            print(f"❌ Error generando embedding: {e}")
            return None
    
    def verify_face(self, face_img):
        """Verifica identidad de un rostro"""
        if not self.embeddings_db:
            return {
                "person": "Sin personas registradas",
                "confidence": 0,
                "verified": False,
                "status": "DB VACÍA"
            }
        
        # Preprocesar rostro
        face_processed = self.preprocess_face(face_img)
        if face_processed is None:
            return {
                "person": "Error procesando",
                "confidence": 0,
                "verified": False,
                "status": "ERROR"
            }
        
        # Generar embedding
        query_embedding = self.generate_embedding(face_processed)
        if query_embedding is None:
            return {
                "person": "Error embedding",
                "confidence": 0,
                "verified": False,
                "status": "ERROR"
            }
        
        # Comparar con todas las personas
        best_match = None
        best_similarity = -1
        
        for person_name, person_data in self.embeddings_db.items():
            # Obtener embedding de referencia
            if isinstance(person_data, dict):
                ref_embedding = np.array(person_data['embedding'])
            else:
                ref_embedding = np.array(person_data)
            
            # Calcular similaridad coseno
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                ref_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_name
        
        # Determinar si está verificado
        is_verified = best_similarity >= self.threshold
        confidence = best_similarity * 100
        
        return {
            "person": best_match if is_verified else "Desconocido",
            "confidence": confidence,
            "similarity": float(best_similarity),
            "verified": is_verified,
            "status": "VERIFICADO" if is_verified else "DESCONOCIDO"
        }
    
    def process_frame_thread(self):
        """Hilo para procesar frames en segundo plano"""
        while self.is_running:
            try:
                # Obtener frame de la cola
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    # Convertir a escala de grises para detección
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Detectar rostros
                    faces = self.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(80, 80)
                    )
                    
                    if len(faces) > 0:
                        # Tomar el rostro más grande
                        faces_sorted = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                        x, y, w, h = faces_sorted[0]
                        
                        # Extraer región del rostro
                        face_img = frame[y:y+h, x:x+w]
                        
                        # Verificar rostro
                        result = self.verify_face(face_img)
                        result['bbox'] = (x, y, w, h)
                        
                        # Agregar resultado a la cola
                        if not self.result_queue.full():
                            self.result_queue.put(result)
                    
                    else:
                        # No se detectaron rostros
                        if not self.result_queue.full():
                            self.result_queue.put({
                                "person": "Sin rostro detectado",
                                "confidence": 0,
                                "verified": False,
                                "status": "BUSCANDO",
                                "bbox": None
                            })
                
                time.sleep(0.05)  # Pequeña pausa para no sobrecargar
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error en hilo de procesamiento: {e}")
    
    def draw_results(self, frame, result):
        """Dibuja resultados en el frame"""
        height, width = frame.shape[:2]
        
        # Colores según resultado
        if result.get('verified', False):
            color = (0, 255, 0)  # Verde para verificado
            bg_color = (0, 200, 0)
        else:
            color = (0, 0, 255)  # Rojo para no verificado
            bg_color = (0, 0, 200)
        
        # Dibujar rectángulo del rostro si existe
        if result.get('bbox'):
            x, y, w, h = result['bbox']
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Etiqueta encima del rostro
            label = f"{result['person']}"
            confidence_text = f"{result['confidence']:.1f}%"
            
            # Calcular posición de texto
            label_size = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)[0]
            conf_size = cv2.getTextSize(confidence_text, self.font, self.font_scale-0.2, self.font_thickness-1)[0]
            
            # Fondo para texto
            cv2.rectangle(frame, (x, y-50), (x+max(label_size[0], conf_size[0])+10, y), bg_color, -1)
            
            # Texto
            cv2.putText(frame, label, (x+5, y-30), self.font, self.font_scale, (255, 255, 255), self.font_thickness)
            cv2.putText(frame, confidence_text, (x+5, y-10), self.font, self.font_scale-0.2, (255, 255, 255), self.font_thickness-1)
        
        # Panel de información en la parte superior
        info_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, info_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Información principal (sin emojis en el estado)
        cv2.putText(frame, f"Estado: {result['status']}", (10, 25), self.font, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Persona: {result['person']}", (10, 50), self.font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Confianza: {result['confidence']:.1f}%", (10, 75), self.font, 0.7, (255, 255, 255), 2)
        
        # Solo mostrar umbral en la parte derecha
        cv2.putText(frame, f"Umbral: {self.threshold}", (width-150, 25), self.font, 0.5, (200, 200, 200), 1)
        
        # Instrucciones
        cv2.putText(frame, "Presiona 'q' para salir, 'r' para reiniciar", (10, height-20), self.font, 0.5, (150, 150, 150), 1)
        cv2.putText(frame, "Presiona '+'/'-' para ajustar umbral", (10, height-5), self.font, 0.5, (150, 150, 150), 1)
        
        return frame
    
    def start_realtime_verification(self, camera_id=0):
        """Inicia verificación facial en tiempo real"""
        print(f"\n🎥 Iniciando verificación en tiempo real...")
        #print(f"📷 Cámara ID: {camera_id}")
        
        # Inicializar cámara
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"❌ No se pudo abrir la cámara {camera_id}")
            return False
        
        # Configurar cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        
        # Iniciar hilo de procesamiento
        self.is_running = True
        process_thread = threading.Thread(target=self.process_frame_thread, daemon=True)
        process_thread.start()
        
        # Resultado por defecto
        current_result = {
            "person": "Iniciando...",
            "confidence": 0,
            "verified": False,
            "status": "INICIANDO",
            "bbox": None
        }
        
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while True:
                # Capturar frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ No se pudo capturar frame")
                    break
                
                # Voltear horizontalmente para efecto espejo
                frame = cv2.flip(frame, 1)
                
                # Agregar frame a cola para procesamiento (no bloquear)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                
                # Obtener último resultado disponible
                while not self.result_queue.empty():
                    try:
                        current_result = self.result_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Dibujar resultados en frame
                frame_with_results = self.draw_results(frame, current_result)
                
                # Calcular FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps_end_time = time.time()
                    fps = 30 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                    print(f"📊 FPS: {fps:.1f}")
                
                # Mostrar frame
                cv2.imshow('Verificación Facial en Tiempo Real', frame_with_results)
                
                # Manejar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("👋 Saliendo...")
                    break
                elif key == ord('r'):
                    print("🔄 Reiniciando detección...")
                    # Limpiar colas
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    while not self.result_queue.empty():
                        try:
                            self.result_queue.get_nowait()
                        except queue.Empty:
                            break
                elif key == ord('+') or key == ord('='):
                    self.threshold = min(1.0, self.threshold + 0.05)
                    print(f"🎯 Nuevo umbral: {self.threshold:.2f}")
                elif key == ord('-'):
                    self.threshold = max(0.1, self.threshold - 0.05)
                    print(f"🎯 Nuevo umbral: {self.threshold:.2f}")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupción por teclado")
        
        except Exception as e:
            print(f"❌ Error durante ejecución: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Limpiar recursos
            self.is_running = False
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Recursos liberados")
        
        return True


def main():
    """Función principal para verificación en tiempo real"""
    print("🎥 VERIFICACIÓN FACIAL EN TIEMPO REAL")
    print("="*50)
    print("🚚 Sistema para Deliverys - Verificación de Personas")
    
    # Configuración
    model_path = input("\n🤖 Ruta del modelo (default: models/face_embeddings_20250617_183416): ").strip()
    if not model_path:
        model_path = "models/face_embeddings_20250617_183416"
    
    # Seleccionar cámara
    camera_id = input("📷 ID de cámara (default: 0): ").strip()
    try:
        camera_id = int(camera_id) if camera_id else 0
    except ValueError:
        camera_id = 0
    
    try:
        # Inicializar sistema
        system = RealtimeFaceVerification(model_path)
        
        if not system.embeddings_db:
            print("\n⚠️ ADVERTENCIA: No hay personas registradas en la base de datos")
            print("📝 Usa el script face_verification_prototype.py para agregar personas primero")
            print("💡 Comando: python face_verification_prototype.py")
            
            continue_anyway = input("\n¿Continuar de todas formas? (y/n): ").strip().lower()
            if continue_anyway != 'y':
                print("👋 Saliendo...")
                return
        
        print(f"\n✅ Sistema inicializado correctamente")
        print(f"👥 Personas registradas: {len(system.embeddings_db)}")
        
        # Iniciar verificación en tiempo real
        system.start_realtime_verification(camera_id)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()