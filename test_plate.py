import numpy as np
from plate_recognition import segment_characters

# Sahte plaka benzeri test görseli oluştur (300x600, 7 karakter)
img = np.zeros((300, 600, 3), dtype=np.uint8)
img[:] = 100  # Gri arka plan

# Beyaz plaka bölgesi
img[120:160, 150:450] = 220

# 7 adet siyah karakter
for i in range(7):
    img[125:155, 160+i*40:160+i*40+20] = 30

p, c, d = segment_characters(img)
print(f"plate_binary shape: {p.shape}")
print(f"characters found: {len(c)}")
print(f"bounds: {d.get('plate_bounds', 'N/A')}")
print(f"col_sums len: {len(d.get('col_sums', []))}")
print("OK")
