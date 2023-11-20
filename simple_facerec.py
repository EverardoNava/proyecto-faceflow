import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        # Lista para almacenar codificaciones de caras conocidas
        self.known_face_encodings = []
        # Lista para almacenar nombres de caras conocidas
        self.known_face_names = []

        # Factor de escalado para redimensionar los marcos de video
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        """
        Carga imágenes con codificación desde la ruta especificada
        :param images_path: Ruta donde se encuentran las imágenes
        :return: None
        """
        # Obtiene la lista de rutas de imágenes en el directorio especificado
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        # Imprime la cantidad de imágenes encontradas
        print("{} encoding images found.".format(len(images_path)))

        # Itera sobre las rutas de las imágenes
        for img_path in images_path:
            # Lee la imagen usando OpenCV
            img = cv2.imread(img_path)
            # Convierte la imagen de formato BGR a RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extrae el nombre de archivo y extensión de la imagen
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            # Obtiene la codificación facial de la primera cara encontrada en la imagen
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Agrega la codificación y el nombre a las listas correspondientes
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        # Imprime un mensaje indicando que las imágenes con codificación han sido cargadas
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        # Redimensiona el marco de video utilizando el factor de escalado
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)

        # Convierte el marco redimensionado de formato BGR a RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Encuentra las ubicaciones de las caras en el marco
        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        # Calcula las codificaciones faciales para las caras detectadas
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Lista para almacenar los nombres de las caras detectadas
        face_names = []
        for face_encoding in face_encodings:
            # Compara las codificaciones faciales detectadas con las codificaciones de caras conocidas
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Calcula las distancias entre las codificaciones faciales detectadas y las de caras conocidas
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            # Encuentra el índice de la mejor coincidencia (la más cercana)
            best_match_index = np.argmin(face_distances)
            
            # Si hay una coincidencia, asigna el nombre correspondiente
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            
            # Agrega el nombre a la lista de nombres de caras
            face_names.append(name)

        # Convierte las ubicaciones de las caras a un array de NumPy
        face_locations = np.array(face_locations)
        # Escala las ubicaciones de las caras de vuelta al tamaño original
        face_locations = face_locations / self.frame_resizing

        # Devuelve las ubicaciones de las caras y los nombres como arrays de NumPy con valores enteros
        return face_locations.astype(int), face_names
