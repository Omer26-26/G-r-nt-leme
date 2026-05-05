import numpy as np
import math

def calculate_histogram(image_matrix):
    """
    0-255 arası her piksel yoğunluğunun frekansını hesaplar.
    Dönüş: Boyutu 256 olan liste veya 1D matris.
    """
    hist = np.zeros(256, dtype=int)
    for val in image_matrix.ravel():
        hist[val] += 1
    return hist

def histogram_stretching(image_matrix):
    """
    Histogram Germe: Görüntüdeki en küçük ve en büyük piksel değerini bulup, 0-255 arasına yayar.
    O_new = (O_old - min) * (255 / (max - min))
    """
    img_min = image_matrix.min()
    img_max = image_matrix.max()
    
    if img_max == img_min:
        return image_matrix.copy()
        
    stretched = (image_matrix - img_min) * (255.0 / (img_max - img_min))
    return np.clip(stretched, 0, 255).astype(np.uint8)

def histogram_equalization(image_matrix):
    """
    Histogram Eşitleme: Kümülatif dağılım fonksiyonu (CDF) kullanarak
    piksel yoğunluk dağılımını eşitler, kontrastı artırır.
    """
    hist = calculate_histogram(image_matrix)
    
    # CDF (Kümülatif Dağılım Fonksiyonu)
    cdf = np.cumsum(hist)
    
    # CDF'in sıfır olmayan ilk değeri
    cdf_min = cdf[cdf > 0].min()
    total = image_matrix.size
    
    # Yeni piksel değerleri: eşitlenmiş histogram
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = round((cdf[i] - cdf_min) / (total - cdf_min) * 255)
    
    equalized = lut[image_matrix]
    return equalized.astype(np.uint8)

def rotate_crop_image(image_matrix, angle_degrees=15):
    """
    Resmi merkezinden belirtilen derece kadar (Nearest Neighbor ile) döndürür, 
    taşan kısımları kırpar (basit crop).
    """
    angle_rad = math.radians(angle_degrees)
    cos_val = math.cos(angle_rad)
    sin_val = math.sin(angle_rad)
    
    img_h, img_w = image_matrix.shape[:2]
    cx, cy = img_w // 2, img_h // 2
    
    out = np.zeros_like(image_matrix)
    is_rgb = len(image_matrix.shape) == 3
    
    for i in range(img_h):
        for j in range(img_w):
            # Merkez etrafında döndür
            offset_x = j - cx
            offset_y = i - cy
            
            src_x = int(offset_x * cos_val + offset_y * sin_val) + cx
            src_y = int(-offset_x * sin_val + offset_y * cos_val) + cy
            
            if 0 <= src_y < img_h and 0 <= src_x < img_w:
                if is_rgb:
                    out[i, j] = image_matrix[src_y, src_x]
                else:
                    out[i, j] = image_matrix[src_y, src_x]
                    
    return out

def zoom_image(image_matrix, scale=1.5):
    """Merkeze odaklı basit manuel yakınlaştırma (Nearest Neighbor)."""
    img_h, img_w = image_matrix.shape[:2]
    
    new_w = int(img_w / scale)
    new_h = int(img_h / scale)
    
    start_x = (img_w - new_w) // 2
    start_y = (img_h - new_h) // 2
    
    cropped = image_matrix[start_y:start_y+new_h, start_x:start_x+new_w]
    
    # Şimdi bu ufak bölgeyi tekrar img_h, img_w boyutuna yayıyoruz
    out = np.zeros_like(image_matrix)
    is_rgb = len(image_matrix.shape) == 3
    
    for i in range(img_h):
        for j in range(img_w):
            src_y = int(i * (new_h / img_h))
            src_x = int(j * (new_w / img_w))
            
            if is_rgb:
                out[i, j] = cropped[src_y, src_x]
            else:
                out[i, j] = cropped[src_y, src_x]
    
    return out
