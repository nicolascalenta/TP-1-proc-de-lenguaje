import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CARGA E INSPECCION ---
img = cv2.imread('Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: No se pudo encontrar la imagen.")
else:
    # ---ESTIRAMIENTO LINEAL (Normalizacion) ---
    img_stretched = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # --- TRANSFORMACION GAMMA ---
    # Como la imagen es muy oscura, usamos un gamma < 1 para aclarar las sombras.
    gamma = 0.4
    # La formula es: Out = ((In / 255) ^ gamma) * 255
    img_gamma = np.array(255 * (img / 255) ** gamma, dtype='uint8')

    # --- ECUALIZACION DE HISTOGRAMA (HE) ---
    # Esta tecnica distribuye las intensidades para que el histograma sea lo mas plano posible.
    img_equ = cv2.equalizeHist(img)

    # --- clahe (Ecualizacion Adaptativa) ---
    # A veces la ecualizacion normal brilla demasiado. clahe es mas sofisticada.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    # --- VISUALIZACION DE RESULTADOS ---
    titles = ['Original', 'Estiramiento (P2)', 'Gamma 0.4 (P3)', 'Ecualización (P4)', 'CLAHE (Extra)']
    images = [img, img_stretched, img_gamma, img_equ, img_clahe]

    plt.figure(figsize=(15, 10))
    for i in range(5):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

        plt.subplot(2, 5, i+6)
        plt.hist(images[i].ravel(), 256, [0, 256])
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()