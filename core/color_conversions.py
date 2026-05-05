import numpy as np

def rgb_to_gray(image_matrix):
    """
    RGB görüntüyü Y = 0.299R + 0.587G + 0.114B formülüyle gri seviyeye çevirir.
    Döngü yerine numpy vektörizasyonu kullanılmıştır.
    """
    if len(image_matrix.shape) == 2:
        return image_matrix # Zaten gri
        
    r = image_matrix[:, :, 0]
    g = image_matrix[:, :, 1]
    b = image_matrix[:, :, 2]
    
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)

def thresholding(image_matrix, threshold_value=127):
    """
    Gri seviye matrisi manuel eşik değeri ile ikili (binary) siyah/beyaz forma çevirir.
    """
    if len(image_matrix.shape) == 3:
        gray = rgb_to_gray(image_matrix)
    else:
        gray = image_matrix.copy()
        
    binary = np.zeros_like(gray)
    binary[gray >= threshold_value] = 255
    return binary.astype(np.uint8)

def contrast_adjustment(image_matrix, alpha=1.5, beta=0):
    """
    Kontrast ayarlama (G_new = alpha * G_old + beta)
    Manuel matris işlemi (Numpy clip ile sınırları kontrol eder).
    """
    adjusted = image_matrix.astype(np.float32) * alpha + beta
    adjusted = np.clip(adjusted, 0, 255)
    return adjusted.astype(np.uint8)

def rgb_to_hsv(image_matrix):
    """
    Manuel RGB → HSV renk uzayı dönüşümü.
    H: 0-180, S: 0-255, V: 0-255
    """
    img = image_matrix.astype(np.float32) / 255.0
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    
    v_max = np.maximum(np.maximum(r, g), b)
    v_min = np.minimum(np.minimum(r, g), b)
    delta = v_max - v_min
    
    # Hue
    h = np.zeros_like(r)
    # delta == 0 → h = 0 (zaten)
    mask_r = (v_max == r) & (delta > 0)
    mask_g = (v_max == g) & (delta > 0)
    mask_b = (v_max == b) & (delta > 0)
    
    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
    
    # Saturation
    s = np.zeros_like(r)
    s[v_max > 0] = delta[v_max > 0] / v_max[v_max > 0]
    
    # Value
    v = v_max
    
    hsv = np.zeros_like(image_matrix)
    hsv[:,:,0] = (h / 2).astype(np.uint8)       # H: 0-180
    hsv[:,:,1] = (s * 255).astype(np.uint8)      # S: 0-255
    hsv[:,:,2] = (v * 255).astype(np.uint8)      # V: 0-255
    return hsv

def rgb_to_ycbcr(image_matrix):
    """
    Manuel RGB → YCbCr renk uzayı dönüşümü.
    Y  =  0.299R + 0.587G + 0.114B
    Cb = -0.169R - 0.331G + 0.500B + 128
    Cr =  0.500R - 0.419G - 0.081B + 128
    """
    img = image_matrix.astype(np.float32)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    
    y  =  0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.500 * b + 128
    cr =  0.500 * r - 0.419 * g - 0.081 * b + 128
    
    ycbcr = np.zeros_like(image_matrix)
    ycbcr[:,:,0] = np.clip(y, 0, 255).astype(np.uint8)
    ycbcr[:,:,1] = np.clip(cb, 0, 255).astype(np.uint8)
    ycbcr[:,:,2] = np.clip(cr, 0, 255).astype(np.uint8)
    return ycbcr

def image_arithmetic(img1, img2, operation='add'):
    """
    İki görüntü arasındaki aritmetik işlemler.
    İkinci görüntü birinciye göre yeniden boyutlandırılır.
    Desteklenen işlemler: add, subtract, multiply, and, or, xor
    """
    # İkinci görüntüyü birincinin boyutuna getir
    if img1.shape != img2.shape:
        h, w = img1.shape[:2]
        from core.geometry_histogram import zoom_image
        # Basit resize: nearest neighbor
        h2, w2 = img2.shape[:2]
        out2 = np.zeros_like(img1)
        for i in range(h):
            for j in range(w):
                src_y = min(int(i * h2 / h), h2 - 1)
                src_x = min(int(j * w2 / w), w2 - 1)
                out2[i, j] = img2[src_y, src_x]
        img2 = out2
    
    a = img1.astype(np.float32)
    b = img2.astype(np.float32)
    
    if operation == 'add':
        result = a + b
    elif operation == 'subtract':
        result = np.abs(a - b)
    elif operation == 'multiply':
        result = (a / 255.0) * b
    elif operation == 'and':
        result = np.bitwise_and(img1, img2).astype(np.float32)
    elif operation == 'or':
        result = np.bitwise_or(img1, img2).astype(np.float32)
    elif operation == 'xor':
        result = np.bitwise_xor(img1, img2).astype(np.float32)
    else:
        result = a
    
    return np.clip(result, 0, 255).astype(np.uint8)

