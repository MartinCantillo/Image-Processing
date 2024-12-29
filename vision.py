import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
image = cv2.imread('C:/Users/marti/OneDrive/Escritorio/VisionArtificial/fotoTest.png', cv2.IMREAD_GRAYSCALE)


# Transformación Lineal
def linear_transformation(image, a=1.2, b=30):
    return np.clip(a * image + b, 0, 255).astype(np.uint8)

# Transformación Logarítmica
def log_transformation(image):
    c = 255 / np.log(1 + np.max(image))
    return (c * (np.log(1 + image))).astype(np.uint8)

# Transformación de Potencia (Raíz)
def power_transformation(image, gamma=0.4):
    c = 255 / (np.max(image) ** gamma)
    return (c * (image ** gamma)).astype(np.uint8)

# Transformación Definida a Trozos
def piecewise_transformation(image):
    result = np.zeros_like(image)
    result[image < 128] = image[image < 128] * 0.5
    result[image >= 128] = 128 + (image[image >= 128] - 128) * 2
    return result.astype(np.uint8)

# Ecualización del Histograma
def histogram_equalization(image):
    return cv2.equalizeHist(image)

# Aplicar las transformaciones
image_linear = linear_transformation(image)
image_log = log_transformation(image)
image_power = power_transformation(image, gamma=0.4)
image_piecewise = piecewise_transformation(image)
image_hist_eq = histogram_equalization(image)

# Mostrar los resultados
titles = ['Original', 'Lineal', 'Logarítmica', 'Potencia', 'Definida a Trozos', 'Ecualización de Histograma']
images = [image, image_linear, image_log, image_power, image_piecewise, image_hist_eq]

plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()