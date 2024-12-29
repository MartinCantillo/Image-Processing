import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar la imagen en escala de grises
image = cv2.imread('C:/Users/marti/OneDrive/Escritorio/VisionArtificial/fotoTest.png', cv2.IMREAD_GRAYSCALE)

# 2. Aplicar un filtro de suavizado (promedio)
kernel = np.ones((5,5), np.float32) / 25  # Definir el kernel 5x5 (matriz de promedio)
smoothed = cv2.filter2D(image, -1, kernel)  # Aplicar el filtro de suavizado a la imagen original

# 3. Aplicar el filtro Sobel para detección de bordes en X y Y
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # Detectar bordes en la dirección X
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Detectar bordes en la dirección Y

# 4. Combinar las imágenes de Sobel en X y Y calculando la magnitud de cada píxel
sobel_combined = sobel_x+ sobel_y  # Combinar bordes X e Y

# 5. Mostrar todas las imágenes en una sola figura
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')  # Ocultar los ejes

plt.subplot(3, 2, 2)
plt.imshow(smoothed, cmap='gray')
plt.title('Filtro de Suavizado')
plt.axis('off')

plt.subplot(3, 2, 3)
plt.imshow(sobel_x, cmap='gray')
plt.title('Filtro Sobel - Eje X')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(sobel_y, cmap='gray')
plt.title('Filtro Sobel - Eje Y')
plt.axis('off')

plt.subplot(3, 2, 5)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Imagen combinada (Sobel X + Sobel Y)')
#plt.axis('off')

plt.tight_layout()  # Ajustar el espacio entre subgráficos
plt.show()
