import numpy as np 
import matplotlib.pyplot as plt

class ImageProcessor:

    def __init__(self):
        
        self.image=None
    @staticmethod #Babalar bu sınıfla uğraşmadan çağırmak için 
    def turn_gray(image):
        if image.ndim == 2:
            return image
        R=image[:,:,0]
        G=image[:,:,1]
        B=image[:,:,2]

        gray=R*0.299+G*0.587+B*0.114 #Ağırlıklı ortalama ile çarptık 
        gray=gray.astype(np.uint8) #Çıkan sayıyı tek matrise çevir

        return gray

    @staticmethod
    def turn_binary(image): #OPSİYONEL Arayüzde eşik değeri değişmek için bir argüman daha eklenebilir fonksiyona 
        threshold=127 #Eşik değeri belirle
        
        if image.ndim==3: #Önce graye çevir
            image=ImageProcessor.turn_gray(image)
        
        binary = (image > threshold).astype(np.uint8) * 255 #True False değerlerini 255 ile çarp Matriste elde et

        return binary    

    @staticmethod
    def stretch_histogram_manual(image):
        """
        cv2.equalizeHist YASAK! 
        Formül: $P_{out} = (P_{in} - min) \times \frac{255}{max - min}$
        """
        # Eğer renkliyse griye çeviriyoruz (çünkü histogram tek kanalda gerilir)
        if image.ndim == 3:
            img_work = ImageProcessor.turn_gray(image)
        else:
            img_work = image.copy()

        img_min = np.min(img_work)
        img_max = np.max(img_work)

        if img_max == img_min:
            return img_work

        # Manuel germe işlemi
        stretched = (img_work - img_min) * (255.0 / (img_max - img_min))
        return stretched.astype(np.uint8)
        
    @staticmethod
    def rgb_to_hsv_manual(image):
        """
        cv2.cvtColor YASAK! Matematiksel HSV dönüşümü.
        """
        # Görüntüyü 0-1 aralığına çekiyoruz (hesaplama kolaylığı için)
        img = image.astype(np.float32) / 255.0
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

        v = np.max(img, axis=2) # Value (Parlaklık)
        m = np.min(img, axis=2) # Minimum değer
        diff = v - m

        # Saturation (Doygunluk)
        s = np.zeros_like(v)
        s[v != 0] = diff[v != 0] / v[v != 0]

        # Hue (Renk Özü) hesaplama
        h = np.zeros_like(v)
        
        # Vektörize edilmiş Hue hesaplaması
        idx = (v == r) & (diff != 0)
        h[idx] = (60 * ((g[idx] - b[idx]) / diff[idx]) + 360) % 360
        
        idx = (v == g) & (diff != 0)
        h[idx] = (60 * ((b[idx] - r[idx]) / diff[idx]) + 120) % 360
        
        idx = (v == b) & (diff != 0)
        h[idx] = (60 * ((r[idx] - g[idx]) / diff[idx]) + 240) % 360

        # Normalizasyon: H (0-179), S (0-255), V (0-255) -> OpenCV standartı için
        h_final = (h / 2).astype(np.uint8)
        s_final = (s * 255).astype(np.uint8)
        v_final = (v * 255).astype(np.uint8)

        return np.stack([h_final, s_final, v_final], axis=2)
    
    @staticmethod
    def resize_manual(image, scale_factor):
        """
        cv2.resize YASAK! En Yakın Komşu (Nearest Neighbor) algoritması ile manuel boyutlandırma.
        """
        old_h, old_w = image.shape[:2]
        new_h = int(old_h * scale_factor)
        new_w = int(old_w * scale_factor)

        # Yeni boyutlar için indeks haritası oluşturuyoruz
        y_indices = (np.arange(new_h) / scale_factor).astype(int)
        x_indices = (np.arange(new_w) / scale_factor).astype(int)

        # Sınır dışına taşmayı önlemek için kırpıyoruz
        y_indices = np.clip(y_indices, 0, old_h - 1)
        x_indices = np.clip(x_indices, 0, old_w - 1)

        # Gelişmiş indeksleme ile yeni resmi oluşturuyoruz
        # Bu yöntem hem 2D hem 3D matrislerde sorunsuz çalışır
        return image[y_indices[:, None], x_indices]
        
    @staticmethod
    def get_histogram(image):
        """
        cv2.calcHist YASAK! Manuel histogram hesaplama.
        """
        if image.ndim == 3:
            image = ImageProcessor.turn_gray(image)
        
        # 0'dan 256'ya kadar bir dizi oluştur (her yoğunluk değeri için bir sayaç)
        hist = np.zeros(256, dtype=int)
        
        # Görüntüyü düzleştir ve her bir değerin kaç kez geçtiğini say
        flat_image = image.ravel()
        for pixel in flat_image:
            hist[pixel] += 1
            
        return hist
    
    @staticmethod
    def plot_histogram(image, title="Histogram"):
        """Histogramı görselleştirmek için eklenen yardımcı fonksiyon."""
        hist = ImageProcessor.get_histogram(image)
        plt.figure()
        plt.title(title)
        plt.bar(range(256), hist, color='gray')
        plt.show()

    ###### Nisa bulanıklaştırma
    @staticmethod
    def turn_blur(image, kernel_size=3):
        # görüntünün kaç satır ve sütundan oluştuğu bilgisini image.shape döndürür.
        height = image.shape[0]
        width = image.shape[1]

        # kenar boşluğunu kernel_size // 2 ile oto. hesaplıyoruz. pad=1 çünkü 3x3 pencere için merkez piksel 1 adım uzakta
        pad = kernel_size // 2

        # sonuç görüntüsü. orijinal görüntüyle aynı boyutta ama içi sıfır
        output = np.zeros_like(image, dtype=np.float64)

        # görüntü renkli mi (3 kanallı) gri mi (2 boyutlu) kontrolüne göre farklı padding uyguluyoruz
        if image.ndim == 3:
            # görüntü renkliyse satır ve sütunlara pad kadar 0 ekliyoruz kanal boyutuna dokunmuyoruz
            padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
        else:
            # görüntü griyse sadece satır ve sütunlara pad ekliyoruz
            padded = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

        # görüntüyü satır satır geziyoruz
        for i in range(height):
            # her satırın her sütununu da tek tek geziyoruz
            for j in range(width):
                window = padded[i : i + kernel_size, j : j + kernel_size]
                output[i, j] = np.mean(window, axis=(0, 1))

        return output.astype(np.uint8)

    # Genişleme
    @staticmethod
    def turn_dilate(image, kernel_size=3):
        # görüntünün kaç satır ve sütundan oluştuğu bilgisi
        height, width = image.shape[:2]
        pad = kernel_size // 2
        output = np.zeros_like(image)
        
        # Boyut kontrolü ile padding
        if image.ndim == 3:
            padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
        else:
            padded = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

        for i in range(height):
            for j in range(width):
                window = padded[i : i + kernel_size, j : j + kernel_size]
                # Pencerede tek bir 255 (beyaz) bile varsa bu piksel 255 olur → beyaz yayılır
                output[i, j] = np.max(window)

        return output.astype(np.uint8)

    # Aşınma
    @staticmethod
    def turn_erode(image, kernel_size=3):
        # görüntünün kaç satır ve sütundan oluştuğu bilgisi
        height, width = image.shape[:2]
        pad = kernel_size // 2
        output = np.zeros_like(image)
        
        # Boyut kontrolü ile padding
        if image.ndim == 3:
            padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=255)
        else:
            padded = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=255)

        for i in range(height):
            for j in range(width):
                window = padded[i : i + kernel_size, j : j + kernel_size]
                # Pencerede tek bir 0 (siyah) bile varsa bu piksel 0 olur → beyaz erir
                output[i, j] = np.min(window)

        return output.astype(np.uint8)

    # Açma
    @staticmethod
    def turn_opening(image, kernel_size=3):
        # Açma = Önce Aşınma, sonra Genişleme
        eroded = ImageProcessor.turn_erode(image, kernel_size)
        return ImageProcessor.turn_dilate(eroded, kernel_size)

    # Kapama
    @staticmethod
    def turn_closing(image, kernel_size=3):
        # Kapama = önce genişleme, sonra aşınma
        dilated = ImageProcessor.turn_dilate(image, kernel_size)
        return ImageProcessor.turn_erode(dilated, kernel_size)

    # Yasin - İki resim arasında aritmetik işlemler
    @staticmethod
    def _prepare_arithmetic(image1, image2):
        """İki resmi aynı boyuta ve tipe getirir."""
        img1 = image1.astype(np.float64)
        h1, w1 = img1.shape[:2]

        # 2. resim yoksa veya boşsa 1. resmi döndür
        if image2 is None:
            return img1, img1

        img2 = image2.astype(np.float64)
        h2, w2 = img2.shape[:2]

        # Boyutlar farklıysa 2. resmi 1. resmin boyutuna tam olarak eşitle (Zorunlu boyutlandırma)
        if h1 != h2 or w1 != w2:
            row_indices = (np.linspace(0, h2 - 1, h1)).astype(int)
            col_indices = (np.linspace(0, w2 - 1, w1)).astype(int)
            if img2.ndim == 3:
                img2 = img2[np.ix_(row_indices, col_indices, [0, 1, 2])]
            else:
                img2 = img2[np.ix_(row_indices, col_indices)]
            img2 = img2.astype(np.float64)

        # Kanal sayılarını eşitle
        if img1.ndim == 3 and img2.ndim == 2:
            # 2. resmi 3 kanala çıkar
            img2 = np.stack([img2, img2, img2], axis=2)
        elif img1.ndim == 2 and img2.ndim == 3:
            # 2. resmi griye çevir
            img2 = ImageProcessor.turn_gray(image2).astype(np.float64)

        return img1, img2

    @staticmethod
    def add_images_manual(image1, image2):
        # Yasin - İki resim arasında aritmetik işlemler: Toplama
        img1, img2 = ImageProcessor._prepare_arithmetic(image1, image2)
        result = img1 + img2
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def multiply_images_manual(image1, image2):
        # Yasin - İki resim arasında aritmetik işlemler: Çarpma
        img1, img2 = ImageProcessor._prepare_arithmetic(image1, image2)
        result = (img1 * img2) / 255.0
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def change_brightness_manual(image, value=30):
        # Yasin - Parlaklık Arttırma
        res = image.astype(np.float64) + value
        return np.clip(res, 0, 255).astype(np.uint8)

    @staticmethod
    def gaussian_blur_manual(image, kernel_size=3, sigma=1.0):
        # Yasin - Konvolüsyon İşlemi (Gauss Bulanıklaştırma) - Optimize Edilmiş Manuel Versiyon
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        gauss = np.exp(-0.5 * (ax**2) / (sigma**2))
        kernel = np.outer(gauss, gauss)
        kernel = kernel / np.sum(kernel)

        pad = kernel_size // 2
        source = image.copy().astype(np.float64)
        h, w = source.shape[:2]

        if source.ndim == 3:
            result = np.zeros_like(source, dtype=np.float64)
            padded = np.pad(source, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
            # Çekirdek boyutuna göre kaydırarak toplama (Vektörize Konvolüsyon)
            for i in range(kernel_size):
                for j in range(kernel_size):
                    result += padded[i:i+h, j:j+w, :] * kernel[i, j]
        else:
            result = np.zeros_like(source, dtype=np.float64)
            padded = np.pad(source, ((pad, pad), (pad, pad)), mode='edge')
            for i in range(kernel_size):
                for j in range(kernel_size):
                    result += padded[i:i+h, j:j+w] * kernel[i, j]

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def adaptive_threshold_manual(image, block_size=15, C=5):
        # Mali - Adaptif eşikleme manuel olarak piksel komşuluk ortalamasıyla uygulanır.
        if block_size % 2 == 0:
            block_size += 1

        if image.ndim == 3:
            gray = ImageProcessor.turn_gray(image)
        else:
            gray = image.copy()

        gray = gray.astype(np.float64)
        height, width = gray.shape
        pad = block_size // 2
        padded = np.pad(gray, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
        output = np.zeros((height, width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                # İç döngü yerine dilimleme kullanarak hızı arttırıyoruz (Mali'nin mantığı aynı kalıyor)
                window = padded[i : i + block_size, j : j + block_size]
                local_mean = np.mean(window)
                threshold = local_mean - C
                output[i, j] = 255 if gray[i, j] > threshold else 0

        return output

    @staticmethod
    def sobel_edge_manual(image, threshold=None):
        # Mali - Sobel kenar bulma manuel olarak Gx ve Gy maskeleriyle uygulanır.
        if image.ndim == 3:
            gray = ImageProcessor.turn_gray(image)
        else:
            gray = image.copy()

        gray = gray.astype(np.float64)
        height, width = gray.shape

        gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

        padded = np.pad(gray, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        magnitude = np.zeros((height, width), dtype=np.float64)

        for i in range(height):
            for j in range(width):
                # 3x3 pencere dilimleme
                window = padded[i : i + 3, j : j + 3]
                gx = np.sum(window * gx_kernel)
                gy = np.sum(window * gy_kernel)
                magnitude[i, j] = np.sqrt(gx * gx + gy * gy)

        max_value = np.max(magnitude)
        if max_value > 0:
            normalized = magnitude * (255.0 / max_value)
        else:
            normalized = magnitude

        if threshold is not None:
            return (normalized >= threshold).astype(np.uint8) * 255

        return normalized.astype(np.uint8)

    @staticmethod
    def add_salt_pepper_noise_manual(image, amount=0.05, seed=None):
        # Mali - Salt & Pepper gürültüsü manuel olarak rastgele pikseller 0 veya 255 yapılarak eklenir.
        amount = max(0.0, min(1.0, amount))
        noisy = image.copy()
        rng = np.random.default_rng(seed)

        height = noisy.shape[0]
        width = noisy.shape[1]
        pepper_limit = amount / 2
        salt_limit = 1 - (amount / 2)

        for i in range(height):
            for j in range(width):
                random_value = rng.random()
                if random_value < pepper_limit:
                    if noisy.ndim == 3:
                        noisy[i, j] = [0, 0, 0]
                    else:
                        noisy[i, j] = 0
                elif random_value > salt_limit:
                    if noisy.ndim == 3:
                        noisy[i, j] = [255, 255, 255]
                    else:
                        noisy[i, j] = 255

        return noisy.astype(np.uint8)

    @staticmethod
    def mean_filter_manual(image, kernel_size=3):
        # Mali - Mean filtre manuel olarak komşuluk penceresindeki piksel ortalamasıyla uygulanır.
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1

        pad = kernel_size // 2
        source = image.copy().astype(np.float64)
        height, width = source.shape[:2]
        result = np.zeros_like(source, dtype=np.float64)

        if source.ndim == 3:
            channel_count = source.shape[2]
            padded = np.pad(source, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            for i in range(height):
                for j in range(width):
                    for channel in range(channel_count):
                        total = 0.0
                        for ki in range(kernel_size):
                            for kj in range(kernel_size):
                                total += padded[i + ki, j + kj, channel]
                        result[i, j, channel] = total / (kernel_size * kernel_size)
        else:
            padded = np.pad(source, ((pad, pad), (pad, pad)), mode="edge")
            for i in range(height):
                for j in range(width):
                    total = 0.0
                    for ki in range(kernel_size):
                        for kj in range(kernel_size):
                            total += padded[i + ki, j + kj]
                    result[i, j] = total / (kernel_size * kernel_size)

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def median_filter_manual(image, kernel_size=3):
        # Mali - Median filtre manuel olarak komşuluk değerleri sıralanıp ortanca değer alınarak uygulanır.
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1

        pad = kernel_size // 2
        source = image.copy()
        height, width = source.shape[:2]
        result = np.zeros_like(source, dtype=np.float64)

        if source.ndim == 3:
            channel_count = source.shape[2]
            padded = np.pad(source, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            for i in range(height):
                for j in range(width):
                    for channel in range(channel_count):
                        values = []
                        for ki in range(kernel_size):
                            for kj in range(kernel_size):
                                values.append(padded[i + ki, j + kj, channel])
                        values.sort()
                        middle_index = len(values) // 2
                        result[i, j, channel] = values[middle_index]
        else:
            padded = np.pad(source, ((pad, pad), (pad, pad)), mode="edge")
            for i in range(height):
                for j in range(width):
                    values = []
                    for ki in range(kernel_size):
                        for kj in range(kernel_size):
                            values.append(padded[i + ki, j + kj])
                    values.sort()
                    middle_index = len(values) // 2
                    result[i, j] = values[middle_index]

        return np.clip(result, 0, 255).astype(np.uint8)


    @staticmethod
    def rgb_to_hsv_manual(image):
        if image.ndim == 2:
            return image
        img = image.astype(np.float32) / 255.0
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        v = np.max(img, axis=2)
        m = np.min(img, axis=2)
        diff = v - m
        s = np.zeros_like(v)
        s[v != 0] = diff[v != 0] / v[v != 0]
        h = np.zeros_like(v)
        idx = (v == r) & (diff != 0)
        h[idx] = (60 * ((g[idx] - b[idx]) / diff[idx]) + 360) % 360
        idx = (v == g) & (diff != 0)
        h[idx] = (60 * ((b[idx] - r[idx]) / diff[idx]) + 120) % 360
        idx = (v == b) & (diff != 0)
        h[idx] = (60 * ((r[idx] - g[idx]) / diff[idx]) + 240) % 360
        h_final = (h / 2).astype(np.uint8)
        s_final = (s * 255).astype(np.uint8)
        v_final = (v * 255).astype(np.uint8)
        return np.stack([h_final, s_final, v_final], axis=2)

    @staticmethod
    def subtract_images_manual(image1, image2):
        # İki resim arasında aritmetik işlemler: Çıkarma
        img1, img2 = ImageProcessor._prepare_arithmetic(image1, image2)
        result = img1 - img2
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def rgb_to_ycbcr_manual(image):
        if image.ndim == 2:
            return image
        # RGB'den YCbCr'ye matematiksel dönüşüm
        img = image.astype(np.float64)
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        return np.stack([y, cb, cr], axis=2).astype(np.uint8)

    @staticmethod
    def adjust_contrast_manual(image, factor=1.5):
        # Kontrast ayarı: (pixel - 128) * factor + 128
        mean = 128
        res = (image.astype(np.float64) - mean) * factor + mean
        return np.clip(res, 0, 255).astype(np.uint8)

    @staticmethod
    def motion_blur_manual(image, kernel_size=15):
        # Motion Blur: Yatay bir çekirdek ile ortalama alma
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel /= kernel_size

        pad = kernel_size // 2
        source = image.copy().astype(np.float64)
        h, w = source.shape[:2]

        if source.ndim == 3:
            result = np.zeros_like(source)
            padded = np.pad(source, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
            for i in range(kernel_size):
                for j in range(kernel_size):
                    if kernel[i, j] > 0:
                        result += padded[i:i+h, j:j+w, :] * kernel[i, j]
        else:
            result = np.zeros_like(source)
            padded = np.pad(source, ((pad, pad), (pad, pad)), mode='edge')
            for i in range(kernel_size):
                for j in range(kernel_size):
                    if kernel[i, j] > 0:
                        result += padded[i:i+h, j:j+w] * kernel[i, j]

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def double_threshold_manual(image, low=50, high=200):
        # Çift Eşikleme
        if image.ndim == 3:
            image = ImageProcessor.turn_gray(image)

        res = np.zeros_like(image)
        res[image >= high] = 255
        res[(image >= low) & (image < high)] = 127
        return res.astype(np.uint8)

    # Arayüze not: Bu her çağrıldığında 90 derece dönüyor

    @staticmethod
    def rotation_image(image):
        if image.ndim == 3:
            transposed = image.transpose(1, 0, 2)  # Transpose alarak sadece eksenlerin yerlerini değiştiriyoruz
        else:
            transposed = image.T  # Binary ve gray için transpose

        rotated = transposed[:, ::-1, ...]  # YÜKSEKLİK: OLDUĞU GİBİ, GENİŞLİK: terse çevir, Kanal: olduğu gibi kalsın
        return rotated

    @staticmethod
    def crop_image(image, x, y, width, height):
        # Gerçek kırpma işlemi: Belirlenen alanı kesip döndürür
        max_h, max_w = image.shape[:2]
        
        # Sınır kontrolleri
        x_end = min(x + width, max_w)
        y_end = min(y + height, max_h)
        
        if image.ndim == 3:
            return image[y:y_end, x:x_end, :].copy()
        else:
            return image[y:y_end, x:x_end].copy()


# --- Test Amaçlı Yardımcı Fonksiyonlar (Arayüzde kullanmayın) ---
def bgr_to_rgb(image):
    return image[:, :, ::-1]  # BGR'I RGB'ye çeviren tersleme işlemim


def rgb_to_bgr(image):
    return image[:, :, ::-1]  # Yukarıdakinin tersi daa


def load_image_cv2(path):
    import cv2
    image = cv2.imread(path)  # Yoldaki görseli okudu, OpenCV olduğu için BGR şeklinde çıktı verdi
    image = bgr_to_rgb(image)  # BGR'yi RGB
    return image


def save_to_image_cv2(image, path):
    import cv2
    output_image = rgb_to_bgr(image)  # Saklamak için bgr'ye çevir
    cv2.imwrite(path, output_image)


def show_image_cv2(image, title="Resim"):
    import cv2
    if image.ndim == 3:
        display_image = rgb_to_bgr(image)  # İMSHOW BGR ÜZERİNDE İŞLEM YAPAR
    else:
        display_image = image

    cv2.imshow(title, display_image)
    cv2.waitKey(0)  # Tuşa basana kadar bekler
    cv2.destroyAllWindows()  # Tuşa basınca kapar