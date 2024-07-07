import cv2
import numpy as np

# Charger le classificateur en cascade de Haar pour la détection de visage
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialiser la caméra (0 est généralement la caméra par défaut)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Utiliser DirectShow

if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la caméra.")
    exit()

while True:
    # Capturer l'image frame par frame
    ret, frame = cap.read()

    if not ret:
        print("Erreur: Impossible de lire le flux vidéo.")
        break

    # Convertir l'image en nuances de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dessiner des rectangles autour des visages détectés
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Afficher l'image avec les visages détectés
    cv2.imshow('Video en Direct - Detection de Visage', frame)

    # Sortir de la boucle en appuyant sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres²
cap.release()
cv2.destroyAllWindows()
