import cv2
import mediapipe as mp
import numpy as np

# Initialiser la caméra
cap = cv2.VideoCapture(0)

# Initialiser Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Couleur et épaisseur de la ligne
line_color = (0, 0, 255)  # Rouge en BGR
line_thickness = 2

# Créer une image vierge pour dessiner
drawing_image = None

# Variables pour stocker la position précédente de l'index
prev_x, prev_y = None, None

if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la caméra.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erreur: Impossible de lire le flux vidéo.")
        break

    # Initialiser l'image de dessin avec les mêmes dimensions que le frame
    if drawing_image is None:
        drawing_image = np.zeros_like(frame)

    # Convertir l'image en RGB pour Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Traiter l'image avec Mediapipe
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Dessiner les landmarks des mains
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Récupérer la position de l'index (landmark 8)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, c = frame.shape
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Si position précédente existe, dessiner une ligne sur l'image de dessin
            if prev_x is not None and prev_y is not None:
                cv2.line(drawing_image, (prev_x, prev_y), (index_x, index_y), line_color, line_thickness)

            # Mettre à jour la position précédente
            prev_x, prev_y = index_x, index_y

    else:
        # Réinitialiser la position si aucune mainq n'est détectée
        prev_x, prev_y = None, None

    # Superposer l'image de dessin au frame original
    combined_frame = cv2.add(frame, drawing_image)

    # Afficher le flux vidéo avec le dessin
    cv2.imshow('Suivi du Doigt Index', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
