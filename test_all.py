"""Tüm modülleri test eder ve hata olup olmadığını gösterir."""
import sys
import traceback
import numpy as np

print("=" * 60)
print("MODUL TEST SCRIPTI")
print("=" * 60)

# 1. utils.io test
print("\n[1] utils.io import...")
try:
    from utils.io import read_image, save_image
    print("   OK")
except Exception as e:
    print(f"   HATA: {e}")
    traceback.print_exc()

# 2. core.color_conversions test
print("\n[2] core.color_conversions import...")
try:
    from core.color_conversions import rgb_to_gray, thresholding, contrast_adjustment
    print("   OK")
except Exception as e:
    print(f"   HATA: {e}")
    traceback.print_exc()

# 3. core.filters test
print("\n[3] core.filters import...")
try:
    from core.filters import manual_convolution2d, apply_mean_filter, apply_median_filter
    print("   OK")
except Exception as e:
    print(f"   HATA: {e}")
    traceback.print_exc()

# 4. core.edge_morphology test
print("\n[4] core.edge_morphology import...")
try:
    from core.edge_morphology import sobel_edge_detection, canny_edge_detection, morph_erosion, morph_dilation
    print("   OK")
except Exception as e:
    print(f"   HATA: {e}")
    traceback.print_exc()

# 5. core.geometry_histogram test
print("\n[5] core.geometry_histogram import...")
try:
    from core.geometry_histogram import calculate_histogram, histogram_stretching, zoom_image, rotate_crop_image
    print("   OK")
except Exception as e:
    print(f"   HATA: {e}")
    traceback.print_exc()

# 6. plate_recognition test
print("\n[6] plate_recognition import...")
try:
    from plate_recognition import segment_characters
    print("   OK")
except Exception as e:
    print(f"   HATA: {e}")
    traceback.print_exc()

# 7. Fonksiyonel Test: Küçük sahte görüntü ile
print("\n" + "=" * 60)
print("FONKSIYONEL TESTLER (50x50 sahte goruntu)")
print("=" * 60)

# 50x50 RGB sahte görüntü
test_img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

print("\n[A] rgb_to_gray...")
try:
    gray = rgb_to_gray(test_img)
    print(f"   OK - shape: {gray.shape}, dtype: {gray.dtype}")
except Exception as e:
    print(f"   HATA: {e}")
    traceback.print_exc()

print("\n[B] thresholding...")
try:
    binary = thresholding(test_img, 127)
    print(f"   OK - shape: {binary.shape}, dtype: {binary.dtype}")
except Exception as e:
    print(f"   HATA: {e}")
    traceback.print_exc()

print("\n[C] contrast_adjustment...")
try:
    cont = contrast_adjustment(test_img, 1.5, 10)
    print(f"   OK - shape: {cont.shape}, dtype: {cont.dtype}")
except Exception as e:
    print(f"   HATA: {e}")
    traceback.print_exc()

print("\n[D] apply_mean_filter (gray)...")
try:
    gray = rgb_to_gray(test_img)
    mf = apply_mean_filter(gray)
    print(f"   OK - shape: {mf.shape}, dtype: {mf.dtype}")
except Exception as e:
    print(f"   HATA: {e}")
    traceback.print_exc()

print("\n[E] apply_median_filter (gray)...")
try:
    gray = rgb_to_gray(test_img)
    medf = apply_median_filter(gray)
    print(f"   OK - shape: {medf.shape}, dtype: {medf.dtype}")
except Exception as e:
    print(f"   HATA: {e}")
    traceback.print_exc()

print("\n[F] sobel_edge_detection...")
try:
    gray = rgb_to_gray(test_img)
    edges = sobel_edge_detection(gray)
    print(f"   OK - shape: {edges.shape}, dtype: {edges.dtype}")
except Exception as e:
    print(f"   HATA: {e}")
    traceback.print_exc()

print("\n[G] segment_characters (LPR pipeline)...")
try:
    binary_res, chars = segment_characters(test_img)
    print(f"   OK - binary shape: {binary_res.shape}, karakter sayisi: {len(chars)}")
except Exception as e:
    print(f"   HATA: {e}")
    traceback.print_exc()

print("\n[H] histogram hesaplama...")
try:
    gray = rgb_to_gray(test_img)
    hist = calculate_histogram(gray)
    print(f"   OK - histogram len: {len(hist)}")
except Exception as e:
    print(f"   HATA: {e}")
    traceback.print_exc()

print("\n[I] histogram_stretching...")
try:
    gray = rgb_to_gray(test_img)
    stretched = histogram_stretching(gray)
    print(f"   OK - shape: {stretched.shape}, dtype: {stretched.dtype}")
except Exception as e:
    print(f"   HATA: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("TESTLER TAMAMLANDI")
print("=" * 60)
