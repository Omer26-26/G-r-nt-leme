import numpy as np
import cv2
from Processor import ImageProcessor  # ImageProcessor sınıfının olduğu dosya adı

def bgr_to_rgb(image):
    return image[:, :, ::-1] #BGR'I RGB'ye çeviren tersleme işlemim

def rgb_to_bgr(image):
    return image[:, :, ::-1] #Yukarıdakinin tersi daa

def load_image(path):

    image=cv2.imread(path) #Yoldaki görseli okudu , OpenCV olduğu için BGR şeklinde çıktı verdi
    image=bgr_to_rgb(image)  # BGR'yi RGB 
    return image

def save_to_image(image,path):

    output_image=rgb_to_bgr(image) #Saklamak için bgr'ye çevircen ömer çünkü imwrite bgr ye göre işlem yapıyor
    image=cv2.imwrite(path,output_image) #okudu çıktısı true false 
    pass

def show_image(image,title="Resim"):  

    if image.ndim==3:
        image=rgb_to_bgr(image) #İMSHOW BGR ÜZERİNDE İŞLEM YAPAR MALİ
        display_image=image

    else : 
        display_image=image # 3 kanallı değilse ya binary ya gray dönüşüme gerek yok
   
    cv2.imshow(title,display_image)
    #Bunları yazdığınız fonksiyonları denemek amaçlı kullanın 
    cv2.waitKey(0) # Tuşa basana kadar bekler
    cv2.destroyAllWindows() # Tuşa basınca kapar
