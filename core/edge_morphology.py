import numpy as np
from core.filters import manual_convolution2d

def sobel_edge_detection(image_matrix):
    """
    Manuel Sobel X ve Y kernelleri kullanarak gradiyent çıkarımı yapar.
    Input: Gri seviye (2D) görüntü.
    Output: G = sqrt(Gx^2 + Gy^2) büyüklük matrisi.
    """
    # X yönü Kernel
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    
    # Y yönü Kernel
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    Gx = manual_convolution2d(image_matrix, Kx)
    Gy = manual_convolution2d(image_matrix, Ky)
    
    G = np.sqrt(Gx**2 + Gy**2)
    max_val = G.max()
    if max_val > 0:
        G = (G / max_val) * 255.0  # Normalize
    
    res = np.zeros_like(image_matrix)
    res[1:-1, 1:-1] = np.clip(G, 0, 255)
    return res.astype(np.uint8)

def morph_erosion(binary_image, kernel_size=3):
    """Görüntü üzerindeki beyaz bölgeleri (255) daraltır (Erosion)."""
    img_h, img_w = binary_image.shape
    pad = kernel_size // 2
    out = np.zeros_like(binary_image)
    
    for i in range(pad, img_h - pad):
        for j in range(pad, img_w - pad):
            region = binary_image[i-pad:i+pad+1, j-pad:j+pad+1]
            if np.all(region == 255):
                out[i, j] = 255
            else:
                out[i, j] = 0
    return out

def morph_dilation(binary_image, kernel_size=3):
    """Görüntü üzerindeki beyaz bölgeleri genişletir (Dilation)."""
    img_h, img_w = binary_image.shape
    pad = kernel_size // 2
    out = np.zeros_like(binary_image)
    
    for i in range(pad, img_h - pad):
        for j in range(pad, img_w - pad):
            region = binary_image[i-pad:i+pad+1, j-pad:j+pad+1]
            if np.any(region == 255):
                out[i, j] = 255
            else:
                out[i, j] = 0
    return out

def morph_opening(binary_image, kernel_size=3):
    """Erosion -> Dilation"""
    eroded = morph_erosion(binary_image, kernel_size)
    return morph_dilation(eroded, kernel_size)

def morph_closing(binary_image, kernel_size=3):
    """Dilation -> Erosion"""
    dilated = morph_dilation(binary_image, kernel_size)
    return morph_erosion(dilated, kernel_size)

def canny_edge_detection(image_matrix):
    """
    Bütün aşamaları birleştirilmiş basit Canny adımları 
    (Sobel->Non Max Compression). Performans için tam Canny yerine sobel+eşik tercih edilebilir 
    fakat temel bir yapı bırakıldı.
    """
    # 1. Blur
    from core.filters import apply_mean_filter
    blurred = apply_mean_filter(image_matrix)
    
    # 2. Sobel
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    Gx = manual_convolution2d(blurred, Kx)
    Gy = manual_convolution2d(blurred, Ky)
    
    G = np.sqrt(Gx**2 + Gy**2)
    max_val = G.max()
    if max_val > 0:
        G = (G / max_val) * 255.0
    
    out = np.zeros_like(image_matrix)
    out[1:-1, 1:-1] = np.clip(G, 0, 255)
    
    # Basite indirgenmiş bir eşikleme yaparak döner
    out[out < 50] = 0
    out[out >= 50] = 255
    return out.astype(np.uint8)
