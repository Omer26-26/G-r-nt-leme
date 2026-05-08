import numpy as np 
import matplotlib.pyplot as plt

class ImageProcessor:

    def __init__(self):
        
        self.image=None
    @staticmethod #Babalar bu sınıfla uğraşmadan çağırmak için 
    def turn_gray(image):

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
        # Hangi yeni piksel, eski resimdeki hangi koordinata denk geliyor?
        row_indices = (np.arange(new_h) / scale_factor).astype(int)
        col_indices = (np.arange(new_w) / scale_factor).astype(int)

        # Sınır dışına taşmayı önlemek için kırpıyoruz
        row_indices = np.clip(row_indices, 0, old_h - 1)
        col_indices = np.clip(col_indices, 0, old_w - 1)

        # Matris dilimleme (Slicing) ile yeni resmi oluşturuyoruz
        if image.ndim == 3:
            return image[np.ix_(row_indices, col_indices, [0, 1, 2])]
        else:
            return image[np.ix_(row_indices, col_indices)]
        
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