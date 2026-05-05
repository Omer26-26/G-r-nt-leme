import numpy as np

def manual_convolution2d(image_matrix, kernel):
    """
    Standart 2D Konvolüsyon İşlemi (Stride=1, Padding=None (Valid)).
    Sadece numpy matris operasyonları kullanır.
    """
    kernel = np.array(kernel)
    k_h, k_w = kernel.shape
    img_h, img_w = image_matrix.shape
    
    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1
    
    output = np.zeros((out_h, out_w), dtype=np.float32)
    
    # 2D kaydırma (sliding) işlemi
    for i in range(out_h):
        for j in range(out_w):
            region = image_matrix[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(region * kernel)
            
    return output

def apply_mean_filter(image_matrix):
    """
    3x3 Ortalama (Mean) Filtresi uygulayarak gürültü azaltır.
    """
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    
    if len(image_matrix.shape) == 3:
        # Renkli ise her kanala ayrı uygula
        out = np.zeros_like(image_matrix)
        for c in range(3):
            ch_out = manual_convolution2d(image_matrix[:, :, c], kernel)
            # Boyut kçüldüğü için, pad ile geri eski boyuta esnetmek veya ortalamak gerekir
            # Basitlik açısından valid convolution sınırlarını alacağız.
            out[1:-1, 1:-1, c] = ch_out.clip(0, 255)
        return out
    else:
        out = manual_convolution2d(image_matrix, kernel)
        res = np.zeros_like(image_matrix)
        res[1:-1, 1:-1] = out.clip(0,255)
        return res.astype(np.uint8)

def apply_median_filter(image_matrix):
    """
    3x3 Median (Ortanca) Filtresi, özellikle Salt&Pepper gürültüsü için.
    """
    img_h, img_w = image_matrix.shape[:2]
    out = np.zeros_like(image_matrix)
    pad = 1
    
    # Eğer renkli ise rgb
    is_rgb = len(image_matrix.shape) == 3
    channels = 3 if is_rgb else 1
    
    for c in range(channels):
        img_ch = image_matrix[:, :, c] if is_rgb else image_matrix
        for i in range(pad, img_h - pad):
            for j in range(pad, img_w - pad):
                region = img_ch[i-pad:i+pad+1, j-pad:j+pad+1]
                val = np.median(region)
                if is_rgb:
                    out[i, j, c] = val
                else:
                    out[i, j] = val
                    
    return out.astype(np.uint8)

def apply_motion_blur(image_matrix, size=5):
    """Yatay düzlemde hareket bulanıklığı efekti verir."""
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    
    if len(image_matrix.shape) == 3:
        out = np.zeros_like(image_matrix)
        pad = size // 2
        for c in range(3):
            ch_out = manual_convolution2d(image_matrix[:,:,c], kernel)
            out[pad:-pad, pad:-pad, c] = ch_out.clip(0,255)
        return out
    else:
        out = manual_convolution2d(image_matrix, kernel)
        res = np.zeros_like(image_matrix)
        pad = size // 2
        res[pad:-pad, pad:-pad] = out.clip(0,255)
        return res.astype(np.uint8)

def add_salt_and_pepper_noise(image_matrix, amount=0.04):
    """Rastgele siyah ve beyaz pikseller (tuz ve biber tuzu) ekler."""
    out = np.copy(image_matrix)
    
    num_salt = np.ceil(amount * image_matrix.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_matrix.shape[:2]]
    
    if len(image_matrix.shape) == 3:
        out[coords[0], coords[1], :] = 255
    else:
        out[coords[0], coords[1]] = 255
        
    num_pepper = np.ceil(amount * image_matrix.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_matrix.shape[:2]]
    
    if len(image_matrix.shape) == 3:
        out[coords[0], coords[1], :] = 0
    else:
        out[coords[0], coords[1]] = 0
        
    return out
