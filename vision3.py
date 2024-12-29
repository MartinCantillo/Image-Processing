import tensorflow as tf
from keras import layers, models
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
image = cv2.imread('C:/Users/marti/OneDrive/Escritorio/VisionArtificial/fotoTest.png', cv2.IMREAD_GRAYSCALE)
image = image.astype('float32') / 255  # Normalizar los valores de los píxeles
image = np.expand_dims(image, axis=(0, -1))  # Añadir dimensión para batch y canales

# Definir el modelo CNN con diferentes filtros
model = models.Sequential()

# Capa convolucional con filtro de suavizado
model.add(layers.Conv2D(1, (5, 5), activation=None, input_shape=(image.shape[1], image.shape[2], 1), padding='same', kernel_initializer=tf.keras.initializers.Constant(value=1/25)))

# Compilar el modelo (no hay pérdida ni métrica ya que solo aplicamos un filtro)
model.compile(optimizer='adam', loss=None)

# Aplicar el filtro a la imagen
filtered_image = model.predict(image)

# Mostrar la imagen original y la imagen filtrada
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(np.squeeze(image), cmap='gray')
plt.title('Imagen Original')

plt.subplot(1, 2, 2)
plt.imshow(np.squeeze(filtered_image), cmap='gray')
plt.title('Imagen Filtrada - CNN (Suavizado)')

plt.show()