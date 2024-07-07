import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

# Charger un modèle pré-entraîné sans la couche de classification finale
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Ajouter des couches de classification personnalisées
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# Créer le modèle final
model = Model(inputs=base_model.input, outputs=predictions)

# Geler les couches du modèle de base pour ne pas les entraîner à nouveau
for layer in base_model.layers:
    layer.trainable = False

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Exemple d'entraînement (données fictives)
import numpy as np
X_train = np.random.rand(100, 224, 224, 3)
y_train = np.random.randint(0, 10, (100, 10))
model.fit(X_train, y_train, epochs=10)
