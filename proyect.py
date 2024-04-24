import cv2
import numpy as np
from keras.models import load_model

# Cargar el modelo
modelo = load_model('emotion_classifier_cnn2.h5')  # Reemplaza 'ruta/a/tu/modelo.h5' con la ruta real de tu modelo



# Cargar el clasificador de rostros de OpenCV
cascada_rostro = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Función para hacer predicciones
def hacer_prediccion(imagen):
    # Convertir la imagen a escala de grises si es en color
    if len(imagen.shape) > 2 and imagen.shape[2] == 3:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen

    # Redimensionar la imagen a 96x96
    imagen_redimensionada = cv2.resize(imagen_gris, (48, 48))

    # Normalizar la imagen para que tenga valores entre 0 y 1
    imagen_redimensionada = imagen_redimensionada / 255.0

    # Agregar una dimensión adicional para el canal (el modelo espera imágenes en formato (batch_size, height, width, channels))
    imagen_redimensionada = np.expand_dims(imagen_redimensionada, axis=-1)

    # Realizar la predicción
    prediccion = modelo.predict(np.expand_dims(imagen_redimensionada, axis=0))

    return prediccion

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

name_classes = ['Angry', 'Fear','Happy', 'Sad', 'Surprise']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a escala de grises si es en color
    if len(frame.shape) > 2 and frame.shape[2] == 3:
        imagen_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = frame

    # Detectar rostros en la imagen
    rostros = cascada_rostro.detectMultiScale(imagen_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterar sobre cada rostro detectado
    for (x, y, w, h) in rostros:
        # Extraer la región de interés (ROI) que contiene el rostro
        roi = imagen_gris[y:y+h, x:x+w]

        # Realizar predicción en el rostro capturado
        resultado_prediccion = hacer_prediccion(roi)
        
        argumento = np.argmax(resultado_prediccion)
        
        result = name_classes[argumento]
        
       # print("---------------",resultado_prediccion)
        
         # Mostrar el resultado de la predicción en la pantalla
        cv2.putText(frame, result, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Dibujar un rectángulo alrededor del rostro detectado
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Muestra el resultado de la predicción en la ventana de OpenCV
        # Aquí puedes agregar código para mostrar el resultado en la ventana de OpenCV
        # por ejemplo, dibujando texto o formas en el frame.

        # Dibujar un rectángulo alrededor del rostro detectado
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Muestra el fotograma con los rostros detectados en la ventana de OpenCV
    cv2.imshow('Captura en tiempo real', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar la ventana de OpenCV
cap.release()
cv2.destroyAllWindows()
