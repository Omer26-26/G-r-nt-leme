import cv2
import numpy as np
import os

def read_image(path):
    """Görüntüyü diskten okur. Türkçe karakterli dizinleri destekler."""
    # cv2.imread unicode (Türkçe) karakterlerde başarısız olur, bu yüzden numpy kullanıyoruz.
    img_array = np.fromfile(path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is not None:
        # OpenCV default BGR, biz RGB olarak işleyeceğiz
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_image(path, image, is_rgb=True):
    """Görüntüyü diske kaydeder (Türkçe karakterleri destekleyerek)."""
    if is_rgb and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    ext = os.path.splitext(path)[1]
    is_success, im_buf_arr = cv2.imencode(ext, image)
    if is_success:
        im_buf_arr.tofile(path)
def show_image_cv2(title, image, is_rgb=True):
    """OpenCV imshow ile gösterme (sadece geliştirme ve test için, GUI'de matplotlib veya PIL kullanılacak)."""
    if is_rgb and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
