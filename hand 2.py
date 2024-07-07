import cv2
import mediapipe as mp

# Initialiser la caméra
cap = cv2.VideoCapture(0)

# Initialiser Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la caméra.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Erreur: Impossible de lire le flux vidéo.")
        break

    # Convertir l'image en RGB pour Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Traiter l'image avec Mediapipe
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Dessiner les landmarks des mains sur le frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Positions des landmarks
            finger_tips = [
                mp_hands.HandLandmark.THUMB_TIP,
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP
            ]
            finger_base = [
                mp_hands.HandLandmark.THUMB_CMC,  # Base de référence pour le pouce
                mp_hands.HandLandmark.INDEX_FINGER_MCP,
                mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                mp_hands.HandLandmark.RING_FINGER_MCP,
                mp_hands.HandLandmark.PINKY_MCP
            ]

            # Compter le nombre de doigts levés
            fingers_up = 0
            for tip, base in zip(finger_tips, finger_base):
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
                    fingers_up += 1

            # Afficher le nombre de doigts levés
            cv2.putText(frame, f'Doigts levés: {fingers_up}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Afficher le flux vidéo avec les dessins
    cv2.imshow('Detection de Main', frame)

    # Quitter la boucle si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
