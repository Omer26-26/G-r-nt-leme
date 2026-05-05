import numpy as np

def segment_characters(image_matrix):
    """
    Geliştirilmiş Plaka Okuma (LPR) Akışı:
    
    1. Gri Seviye Dönüşüm
    2. Mean Filter (Gürültü Azaltma)
    3. Sobel Edge Detection (Gx, Gy, G)
    4. Manuel Eşikleme (Binarization)
    5. Morfolojik Kapama (Karakter bölgelerini birleştirme)
    6. Yatay/Dikey Projeksiyon + En Boy Oranı ile Plaka Bölgesi Tespiti
    7. Plaka Kırpma + Otsu Thresholding
    8. Dikey Projeksiyon ile Karakter Segmentasyonu
    """
    from core.color_conversions import rgb_to_gray, thresholding
    from core.filters import apply_mean_filter, manual_convolution2d

    # ─── ADIM 1: Gri Seviye ───
    gray = rgb_to_gray(image_matrix) if len(image_matrix.shape) == 3 else image_matrix.copy()
    
    # ─── ADIM 2: Mean Filter ───
    filtered = apply_mean_filter(gray)
    
    # ─── ADIM 3: Sobel Edge Detection ───
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    Gx_raw = manual_convolution2d(filtered, Kx)
    Gy_raw = manual_convolution2d(filtered, Ky)
    G_raw = np.sqrt(Gx_raw**2 + Gy_raw**2)
    
    def _norm255(m):
        mx = m.max()
        if mx > 0:
            return ((m / mx) * 255).astype(np.uint8)
        return m.astype(np.uint8)
    
    sobel_gx = np.zeros_like(gray)
    sobel_gx[1:-1, 1:-1] = _norm255(np.abs(Gx_raw))
    
    sobel_gy = np.zeros_like(gray)
    sobel_gy[1:-1, 1:-1] = _norm255(np.abs(Gy_raw))
    
    edges = np.zeros_like(gray)
    edges[1:-1, 1:-1] = _norm255(G_raw)
    
    # ─── ADIM 4: Manuel Eşikleme ───
    edge_binary = thresholding(edges, 80)
    
    # ─── ADIM 5-6: Plaka Bölgesi Tespiti (Kenar tabanlı) ───
    plate_y1, plate_y2, plate_x1, plate_x2 = _find_plate_region(edge_binary)
    
    # ─── ADIM 6.5: Kenarlara Göre Daraltma (Tightening) ───
    # Bulunan aday bölgenin içindeki kenar piksellerinin sınırlarına tam oturacak şekilde daralt
    plate_y1, plate_y2, plate_x1, plate_x2 = _tighten_bounds(
        edge_binary, plate_y1, plate_y2, plate_x1, plate_x2
    )
    
    # ─── ADIM 7: Plaka Kırpma + Otsu ───
    plate_region_gray = gray[plate_y1:plate_y2, plate_x1:plate_x2]
    
    if plate_region_gray.size == 0:
        return gray, [], {}
    
    plate_thresh = _otsu_threshold(plate_region_gray)
    plate_binary = thresholding(plate_region_gray, plate_thresh)
    
    # Plakalar: beyaz zemin + siyah karakter
    white_ratio = np.sum(plate_binary == 255) / plate_binary.size
    if white_ratio < 0.3:
        plate_binary = 255 - plate_binary
    
    # ─── ADIM 8: Dikey Projeksiyon ile Karakter Segmentasyonu ───
    characters, col_sums = _vertical_projection_segment(plate_binary)
    
    # Debug bilgileri
    debug_info = {
        'gray': gray,
        'filtered': filtered,
        'sobel_gx': sobel_gx,
        'sobel_gy': sobel_gy,
        'edges': edges,
        'edge_binary': edge_binary,
        'plate_bounds': (plate_y1, plate_y2, plate_x1, plate_x2),
        'plate_gray': plate_region_gray,
        'plate_binary': plate_binary,
        'plate_thresh_value': plate_thresh,
        'col_sums': col_sums,
    }
    
    return plate_binary, characters, debug_info


# ═══════════════════════════════════════════════════════════
#  PLAKA BÖLGESI TESPİTİ
# ═══════════════════════════════════════════════════════════

def _find_plate_region(edge_binary):
    """
    Birden fazla aday bölge bulur ve en iyi olanı seçer.
    
    Türk plakaları: ~520mm × 110mm → en boy oranı ≈ 4.7
    Kabul aralığı: 2.0 - 7.0
    
    Algoritma:
    1. Yatay projeksiyon → yoğun satır bantlarını bul
    2. Her bant için dikey projeksiyon → sütun aralığını bul
    3. Her aday bölge için skor hesapla (en boy oranı + kenar yoğunluğu)
    4. En iyi skoru döndür
    """
    h, w = edge_binary.shape
    
    # ── Adım 1: Yatay projeksiyon ──
    row_sums = np.sum(edge_binary == 255, axis=1)
    
    # Düzleştirme
    win = max(3, h // 40)
    smoothed = np.convolve(row_sums, np.ones(win)/win, mode='same')
    
    # ── Adım 2: Yoğun bantları bul ──
    # Global ortalamadan yüksek olan satırları bul
    mean_val = smoothed.mean()
    if mean_val == 0:
        return 0, h, 0, w
    
    candidates = []
    
    # Farklı eşik seviyeleriyle bantları tara
    for threshold_ratio in [0.6, 0.5, 0.4, 0.3]:
        threshold = smoothed.max() * threshold_ratio
        bands = _find_bands(smoothed, threshold, min_gap=3)
        
        for band_y1, band_y2 in bands:
            band_h = band_y2 - band_y1
            
            # Bant yüksekliği makul mü? (toplam yüksekliğin %2-40'ı)
            if band_h < h * 0.02 or band_h > h * 0.45:
                continue
            
            # ── Adım 3: Bu bant için dikey projeksiyon ──
            strip = edge_binary[band_y1:band_y2, :]
            col_sums = np.sum(strip == 255, axis=0)
            
            col_win = max(3, w // 50)
            col_smooth = np.convolve(col_sums, np.ones(col_win)/col_win, mode='same')
            
            col_thresh = col_smooth.max() * 0.25
            col_bands = _find_bands(col_smooth, col_thresh, min_gap=3)
            
            for band_x1, band_x2 in col_bands:
                band_w = band_x2 - band_x1
                
                if band_w < w * 0.05:
                    continue
                
                # ── En boy oranı kontrolü ──
                aspect_ratio = band_w / max(band_h, 1)
                
                # Cömert margin ekle (Böylece bölge biraz geniş tutulur)
                margin_y = max(4, int(band_h * 0.25))
                margin_x = max(4, int(band_w * 0.10))
                
                y1 = max(0, band_y1 - margin_y)
                y2 = min(h, band_y2 + margin_y)
                x1 = max(0, band_x1 - margin_x)
                x2 = min(w, band_x2 + margin_x)
                
                # Skor hesapla
                score = _score_candidate(edge_binary, y1, y2, x1, x2, aspect_ratio)
                
                if score > 0:
                    candidates.append((y1, y2, x1, x2, score))
    
    if not candidates:
        # Hiç aday bulunamadıysa, en yoğun bölgeyi basitçe al
        return _fallback_detection(edge_binary)
    
    # En iyi adayı seç
    candidates.sort(key=lambda c: c[4], reverse=True)
    best = candidates[0]
    
    return best[0], best[1], best[2], best[3]


def _find_bands(signal, threshold, min_gap=3):
    """
    1D sinyalde eşik üzerindeki bitişik bölgeleri (bantları) bulur.
    Küçük boşlukları (min_gap) birleştirir.
    """
    above = signal > threshold
    bands = []
    in_band = False
    start = 0
    last_end = 0
    
    for i in range(len(above)):
        if above[i] and not in_band:
            # Önceki bant ile aradaki boşluk çok küçükse birleştir
            if bands and (i - last_end) <= min_gap:
                start = bands[-1][0]
                bands.pop()
            else:
                start = i
            in_band = True
        elif not above[i] and in_band:
            in_band = False
            last_end = i
            bands.append((start, i))
    
    if in_band:
        bands.append((start, len(signal)))
    
    return bands


def _score_candidate(edge_binary, y1, y2, x1, x2, aspect_ratio):
    """
    Bir aday plaka bölgesini puanlar.
    
    Yüksek puan = plaka olma olasılığı yüksek
    - En boy oranı 4.7'ye yakınsa yüksek puan
    - Bölge içindeki kenar yoğunluğu yüksekse yüksek puan
    - Çok küçük veya çok büyük bölgeler düşük puan
    """
    h, w = edge_binary.shape
    region_h = y2 - y1
    region_w = x2 - x1
    area = region_h * region_w
    
    if area <= 0:
        return 0
    
    # En boy oranı skoru (4.7'ye yakınlık, 2-7 arası kabul)
    if aspect_ratio < 1.5 or aspect_ratio > 8.0:
        return 0  # Kesinlikle plaka değil
    
    # Gaussian benzeri skor: 4.7 merkezli
    ar_score = np.exp(-((aspect_ratio - 4.7) ** 2) / (2 * 1.5**2))
    
    # Kenar yoğunluğu skoru
    region = edge_binary[y1:y2, x1:x2]
    edge_density = np.sum(region == 255) / area
    density_score = min(edge_density * 5, 1.0)  # 0-1 arası normalize
    
    # Boyut skoru (çok küçük veya çok büyük olmamalı)
    total_area = h * w
    area_ratio = area / total_area
    if area_ratio < 0.001 or area_ratio > 0.5:
        return 0
    size_score = 1.0 - abs(area_ratio - 0.05) * 5  # %5 civarı ideal
    size_score = max(0, min(1, size_score))
    
    # Toplam skor
    total_score = ar_score * 0.5 + density_score * 0.35 + size_score * 0.15
    
    return total_score


def _fallback_detection(edge_binary):
    """
    Aday bulunamazsa basit bir fallback: en yoğun bölgeyi al.
    """
    h, w = edge_binary.shape
    
    row_sums = np.sum(edge_binary == 255, axis=1)
    win = max(3, h // 30)
    smoothed_r = np.convolve(row_sums, np.ones(win)/win, mode='same')
    
    peak_r = np.argmax(smoothed_r)
    band_h = max(h // 8, 20)
    y1 = max(0, peak_r - band_h // 2)
    y2 = min(h, peak_r + band_h // 2)
    
    strip = edge_binary[y1:y2, :]
    col_sums = np.sum(strip == 255, axis=0)
    win_c = max(3, w // 40)
    smoothed_c = np.convolve(col_sums, np.ones(win_c)/win_c, mode='same')
    
    peak_c = np.argmax(smoothed_c)
    band_w = max(w // 3, 50)
    x1 = max(0, peak_c - band_w // 2)
    x2 = min(w, peak_c + band_w // 2)
    
    return y1, y2, x1, x2


def _tighten_bounds(edge_binary, y1, y2, x1, x2):
    """
    Bulunan aday bölge içindeki en dış beyaz (kenar) piksellerinin sınırlarını bularak bölgeyi tam sıfıra sıfır oturtur.
    Kırpılan bölgedeki gürültü boşluklarını atar.
    """
    region = edge_binary[y1:y2, x1:x2]
    
    # Beyaz piksel olan satır ve sütunlar
    rows_with_edges = np.any(region == 255, axis=1)
    cols_with_edges = np.any(region == 255, axis=0)
    
    if not np.any(rows_with_edges) or not np.any(cols_with_edges):
        return y1, y2, x1, x2
    
    # İlk ve son beyaz piksellerin lokal koordinatları
    ymin, ymax = np.where(rows_with_edges)[0][[0, -1]]
    xmin, xmax = np.where(cols_with_edges)[0][[0, -1]]
    
    # Global koordinatlara çevir
    new_y1 = y1 + ymin
    new_y2 = y1 + ymax + 1
    new_x1 = x1 + xmin
    new_x2 = x1 + xmax + 1
    
    # Karakterlerin kenarları çok sátır sınırında kalmasın diye ufak bir padding ekle
    pad_y = 4
    pad_x = 4
    
    h, w = edge_binary.shape
    new_y1 = max(0, new_y1 - pad_y)
    new_y2 = min(h, new_y2 + pad_y)
    new_x1 = max(0, new_x1 - pad_x)
    new_x2 = min(w, new_x2 + pad_x)
    
    return new_y1, new_y2, new_x1, new_x2


# ═══════════════════════════════════════════════════════════
#  OTSU EŞİKLEME
# ═══════════════════════════════════════════════════════════

def _otsu_threshold(gray_image):
    """
    Manuel Otsu eşikleme: Sınıflar-arası varyansı maksimize eden eşik değerini bulur.
    (Raporda Otsu (1979) referansına uygun)
    """
    hist = np.zeros(256, dtype=int)
    for val in gray_image.ravel():
        hist[int(val)] += 1
    
    total = gray_image.size
    sum_total = np.sum(np.arange(256) * hist)
    
    sum_bg = 0
    weight_bg = 0
    max_variance = 0
    best_threshold = 127
    
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        
        sum_bg += t * hist[t]
        
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        
        if variance > max_variance:
            max_variance = variance
            best_threshold = t
    
    return best_threshold


# ═══════════════════════════════════════════════════════════
#  KARAKTER SEGMENTASYONU
# ═══════════════════════════════════════════════════════════

def _vertical_projection_segment(plate_binary):
    """
    Dikey Projeksiyon ile karakter segmentasyonu.
    Kırpılmış ve binary plaka matrisi üzerinde çalışır.
    
    Returns:
        characters: Karakter alt matrisleri listesi
        col_sums: Sütun bazlı siyah piksel sayıları (grafik için)
    """
    h, w = plate_binary.shape
    
    # Siyah piksellerin (karakter) sütun bazlı sayısı
    col_sums = np.sum(plate_binary == 0, axis=0)
    
    # Eşik: sütundaki siyah piksel, yüksekliğin %5'inden fazla olmalı
    threshold_sum = max(2, int(h * 0.05))
    
    # Segmentleri bul
    in_char = False
    start_c = 0
    segments = []
    
    for c in range(w):
        if col_sums[c] > threshold_sum and not in_char:
            in_char = True
            start_c = c
        elif col_sums[c] <= threshold_sum and in_char:
            in_char = False
            end_c = c
            char_w = end_c - start_c
            if char_w >= max(3, int(h * 0.05)):
                segments.append((start_c, end_c))
    
    # Son karakter
    if in_char:
        end_c = w
        char_w = end_c - start_c
        if char_w >= max(3, int(h * 0.05)):
            segments.append((start_c, end_c))
    
    # Birbirine çok yakın segmentleri birleştir
    merged = []
    for seg in segments:
        if merged and (seg[0] - merged[-1][1]) < max(2, int(h * 0.03)):
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)
    segments = merged
    
    # Aşırı geniş veya dar segmentleri filtrele
    if len(segments) >= 3:
        widths = [s[1] - s[0] for s in segments]
        median_w = sorted(widths)[len(widths) // 2]
        min_allowed = max(3, median_w * 0.25)
        max_allowed = median_w * 3.0
        segments = [s for s in segments if min_allowed <= (s[1] - s[0]) <= max_allowed]
    
    # Karakter alt matrislerini çıkar (dikey kırpma ile)
    characters = []
    for start_c, end_c in segments:
        char_slice = plate_binary[:, start_c:end_c]
        
        # Siyah piksel içeren satır aralığını bul
        row_has_black = np.sum(char_slice == 0, axis=1) > 0
        active_rows = np.where(row_has_black)[0]
        
        if len(active_rows) > 0:
            ry1 = max(0, active_rows[0] - 2)
            ry2 = min(h, active_rows[-1] + 3)
            char_cropped = char_slice[ry1:ry2, :]
            if char_cropped.size > 0 and char_cropped.shape[0] > 3 and char_cropped.shape[1] > 2:
                characters.append(char_cropped)
    
    return characters, col_sums
