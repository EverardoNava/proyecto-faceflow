import cv2
from simple_facerec import SimpleFacerec
from datetime import datetime  # Importa el módulo datetime

# Inicializa el objeto SimpleFacerec, que se utilizará para el reconocimiento facial
sfr = SimpleFacerec()

# Carga las imágenes de referencia para el reconocimiento facial desde la carpeta "images/"
sfr.load_encoding_images("images/")

# Inicializa la captura de video desde la cámara (puede ser la cámara integrada o una cámara externa)
cap = cv2.VideoCapture(0)

output_file = open("reconocimientos.txt", "w")

# Bucle principal para procesar continuamente los frames de video
# Bucle principal para procesar continuamente los frames de video
while True:
    # Lee un frame desde la cámara
    ret, frame = cap.read()

    # Verifica que el frame no esté vacío
    if not ret or frame is None:
        break

    # Detecta caras en el frame y obtiene las ubicaciones y nombres de las caras conocidas
    face_locations, face_names = sfr.detect_known_faces(frame)


    # Itera sobre las caras detectadas y sus nombres asociados
    for face_loc, name in zip(face_locations, face_names):
        # Extrae las coordenadas de la cara (esquina superior izquierda y esquina inferior derecha)
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Dibuja un rectángulo alrededor de la cara
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        # Obtiene la hora actual
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Muestra el nombre asociado a la cara en una línea
        cv2.putText(frame, name, (x1, y1 - 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

        # Muestra la hora en una línea separada debajo del nombre
        cv2.putText(frame, current_time, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        
        # Escribe el nombre de la persona en el archivo de texto junto con la hora
        output_file.write(f"{name} - {current_time}\n")





    

    # Muestra el frame resultante con las caras detectadas y los nombres
    cv2.imshow("Face Flow", frame)

    # Espera la tecla 'Esc' (27 en la codificación ASCII) para salir del bucle
    key = cv2.waitKey(1)
    if key == 27:
        break

# Libera los recursos de la cámara
cap.release()

# Cierra todas las ventanas abiertas
cv2.destroyAllWindows()
