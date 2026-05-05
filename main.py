import customtkinter as ctk
from PIL import Image, ImageTk
import tkinter.filedialog as fd
import tkinter.messagebox as mb
import numpy as np
import threading
import traceback

from utils.io import read_image, save_image
from core.color_conversions import rgb_to_gray, thresholding, contrast_adjustment, rgb_to_hsv, rgb_to_ycbcr, image_arithmetic
from core.filters import apply_mean_filter, apply_median_filter, apply_motion_blur, add_salt_and_pepper_noise
from core.edge_morphology import sobel_edge_detection, canny_edge_detection, morph_erosion, morph_dilation, morph_opening, morph_closing
from core.geometry_histogram import zoom_image, rotate_crop_image, calculate_histogram, histogram_stretching, histogram_equalization
from plate_recognition import segment_characters

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Görüntü İşleme Projesi (Numpy ile Sıfırdan)")
        self.geometry("1200x750")
        self.minsize(900, 600)

        self.original_image_matrix = None
        self.current_image_matrix = None
        self.second_image_matrix = None  # İki görüntü arası aritmetik işlemler için
        
        # CTkImage referanslarını sakla (garbage collection'dan korumak için)
        self._ctk_img_orig = None
        self._ctk_img_mod = None

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.pack(side="left", fill="y", padx=0, pady=0)
        self.sidebar_frame.pack_propagate(False)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, text="🖼️ Manuel Filtreler",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.pack(padx=20, pady=(20, 10))

        self.btn_load = ctk.CTkButton(
            self.sidebar_frame, text="📂 Resim Yükle",
            command=self.load_image, fg_color="#2563EB", hover_color="#1D4ED8"
        )
        self.btn_load.pack(padx=20, pady=10, fill="x")

        self.btn_save = ctk.CTkButton(
            self.sidebar_frame, text="💾 Resmi Kaydet",
            command=self.save_current, fg_color="#059669", hover_color="#047857"
        )
        self.btn_save.pack(padx=20, pady=5, fill="x")

        self.btn_load2 = ctk.CTkButton(
            self.sidebar_frame, text="📂 2. Resim Yükle (Aritmetik)",
            command=self.load_second_image, fg_color="#7C3AED", hover_color="#6D28D9",
            font=ctk.CTkFont(size=11)
        )
        self.btn_load2.pack(padx=20, pady=5, fill="x")

        # --- Filtre Seçimi ---
        ctk.CTkLabel(self.sidebar_frame, text="Filtre / İşlem Seçin:",
                     font=ctk.CTkFont(size=13)).pack(padx=20, pady=(20, 5))

        self.filter_option = ctk.CTkComboBox(self.sidebar_frame, values=[
            "1. Gri Seviye",
            "2. Thresholding (Eşikleme)",
            "3. Kontrast Ayarı",
            "4. Histogram Germe",
            "5. Histogram Eşitleme",
            "6. Histogram Görüntüle",
            "7. RGB → HSV Dönüşümü",
            "8. RGB → YCbCr Dönüşümü",
            "9. Döndür (30°)",
            "10. Zoom (1.5x)",
            "11. Salt & Pepper Gürültüsü",
            "12. Mean Filtre",
            "13. Median Filtre",
            "14. Motion Blur",
            "15. Sobel Edge",
            "16. Canny (Sobel+Eşik)",
            "17. Morph Erosion (Aşınma)",
            "18. Morph Dilation (Genişleme)",
            "19. Morph Opening (Açma)",
            "20. Morph Closing (Kapama)",
            "21. Aritmetik: Toplama",
            "22. Aritmetik: Çıkarma",
            "23. Aritmetik: AND",
            "24. Aritmetik: OR",
            "25. Aritmetik: XOR",
            "26. PLAKA OKUMA (LPR)"
        ], width=200)
        self.filter_option.pack(padx=20, pady=5, fill="x")

        self.btn_apply = ctk.CTkButton(
            self.sidebar_frame, text="▶ Uygula",
            command=self.apply_filter, fg_color="#DC2626", hover_color="#B91C1C"
        )
        self.btn_apply.pack(padx=20, pady=10, fill="x")

        self.btn_reset = ctk.CTkButton(
            self.sidebar_frame, text="↺ Sıfırla",
            command=self.reset_image, fg_color="#6B7280", hover_color="#4B5563"
        )
        self.btn_reset.pack(padx=20, pady=5, fill="x")

        # --- Durum Etiketi ---
        self.status_label = ctk.CTkLabel(
            self.sidebar_frame, text="Durum: Hazır",
            font=ctk.CTkFont(size=11), text_color="#9CA3AF"
        )
        self.status_label.pack(padx=20, pady=(20, 5), side="bottom")

        # --- Progress Bar ---
        self.progress = ctk.CTkProgressBar(self.sidebar_frame, mode="indeterminate")
        self.progress.pack(padx=20, pady=(0, 10), fill="x", side="bottom")
        self.progress.set(0)

        # --- Main Image Area ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.label_title = ctk.CTkLabel(
            self.main_frame,
            text="Orijinal vs İşlenmiş Görüntü",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.label_title.pack(pady=10)

        self.image_container = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.image_container.pack(expand=True, fill="both", padx=10, pady=10)

        # Sol panel - Orijinal
        left_frame = ctk.CTkFrame(self.image_container)
        left_frame.pack(side="left", padx=5, expand=True, fill="both")
        ctk.CTkLabel(left_frame, text="Orijinal", font=ctk.CTkFont(size=12, weight="bold")).pack(pady=5)
        self.panel_orig = ctk.CTkLabel(left_frame, text="Resim yüklenmedi\n\n📂 Sol menüden\n'Resim Yükle' butonuna basın")
        self.panel_orig.pack(expand=True, padx=10, pady=10)

        # Sağ panel - İşlenmiş
        right_frame = ctk.CTkFrame(self.image_container)
        right_frame.pack(side="right", padx=5, expand=True, fill="both")
        ctk.CTkLabel(right_frame, text="İşlenmiş", font=ctk.CTkFont(size=12, weight="bold")).pack(pady=5)
        self.panel_mod = ctk.CTkLabel(right_frame, text="Henüz işlem yapılmadı\n\nBir filtre seçip\n'Uygula' butonuna basın")
        self.panel_mod.pack(expand=True, padx=10, pady=10)

    # ------------------------------------------------------------------
    def set_status(self, msg):
        self.status_label.configure(text=f"Durum: {msg}")
        self.update_idletasks()

    def load_image(self):
        filepath = fd.askopenfilename(
            title="Görüntü Dosyası Seçin",
            filetypes=[
                ("Tüm Görüntüler", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"),
                ("BMP", "*.bmp"), ("Tüm Dosyalar", "*.*")
            ]
        )
        if not filepath:
            return

        self.set_status("Yükleniyor...")
        try:
            img = read_image(filepath)
            if img is None:
                mb.showerror("Hata", f"Görüntü okunamadı!\n\nDosya: {filepath}")
                self.set_status("Yükleme başarısız")
                return

            self.original_image_matrix = img
            self.current_image_matrix = img.copy()
            self.display_images()
            h, w = img.shape[:2]
            self.set_status(f"Yüklendi ({w}x{h})")
        except Exception as e:
            mb.showerror("Hata", f"Dosya yüklenirken hata:\n{e}")
            traceback.print_exc()
            self.set_status("Hata!")

    def load_second_image(self):
        """İki görüntü arasındaki aritmetik işlemler için 2. resmi yükler."""
        filepath = fd.askopenfilename(
            title="2. Görüntü Dosyası Seçin (Aritmetik İşlemler İçin)",
            filetypes=[
                ("Tüm Görüntüler", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        if not filepath:
            return
        try:
            img = read_image(filepath)
            if img is None:
                mb.showerror("Hata", f"2. görüntü okunamadı!\n{filepath}")
                return
            self.second_image_matrix = img
            self.set_status(f"2. resim yüklendi ({img.shape[1]}x{img.shape[0]})")
        except Exception as e:
            mb.showerror("Hata", f"2. görüntü yüklenirken hata:\n{e}")

    def save_current(self):
        if self.current_image_matrix is None:
            mb.showwarning("Uyarı", "Kaydedilecek görüntü yok!")
            return
        filepath = fd.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")]
        )
        if filepath:
            try:
                is_rgb = len(self.current_image_matrix.shape) == 3
                save_image(filepath, self.current_image_matrix, is_rgb=is_rgb)
                self.set_status("Kaydedildi ✓")
            except Exception as e:
                mb.showerror("Hata", f"Kaydetme hatası:\n{e}")

    def reset_image(self):
        if self.original_image_matrix is not None:
            self.current_image_matrix = self.original_image_matrix.copy()
            self.display_images()
            self.set_status("Sıfırlandı ✓")

    def display_images(self):
        max_size = (450, 400)
        
        if self.original_image_matrix is not None:
            try:
                orig = self.original_image_matrix
                if orig.dtype != np.uint8:
                    orig = np.clip(orig, 0, 255).astype(np.uint8)
                if len(orig.shape) == 2:
                    pil_o = Image.fromarray(orig, mode='L')
                else:
                    pil_o = Image.fromarray(orig)
                pil_o.thumbnail(max_size, Image.Resampling.LANCZOS)
                self._ctk_img_orig = ctk.CTkImage(
                    light_image=pil_o, dark_image=pil_o,
                    size=(pil_o.size[0], pil_o.size[1])
                )
                self.panel_orig.configure(image=self._ctk_img_orig, text="")
            except Exception as e:
                print(f"Orijinal görüntü gösterim hatası: {e}")
                traceback.print_exc()

        if self.current_image_matrix is not None:
            try:
                curr = self.current_image_matrix
                if curr.dtype != np.uint8:
                    curr = np.clip(curr, 0, 255).astype(np.uint8)
                if len(curr.shape) == 2:
                    pil_m = Image.fromarray(curr, mode='L')
                else:
                    pil_m = Image.fromarray(curr)
                pil_m.thumbnail(max_size, Image.Resampling.LANCZOS)
                self._ctk_img_mod = ctk.CTkImage(
                    light_image=pil_m, dark_image=pil_m,
                    size=(pil_m.size[0], pil_m.size[1])
                )
                self.panel_mod.configure(image=self._ctk_img_mod, text="")
            except Exception as e:
                print(f"İşlenmiş görüntü gösterim hatası: {e}")
                traceback.print_exc()

    # ------------------------------------------------------------------
    def apply_filter(self):
        if self.current_image_matrix is None:
            mb.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return

        choice = self.filter_option.get()
        self.set_status(f"İşleniyor: {choice}...")
        self.progress.start()
        self.btn_apply.configure(state="disabled")

        # İşlemi ayrı thread'de çalıştır (GUI donmasın)
        thread = threading.Thread(target=self._run_filter, args=(choice,), daemon=True)
        thread.start()

    def _run_filter(self, choice):
        try:
            mat = self.current_image_matrix.copy()
            result = None

            if "Gri Seviye" in choice:
                result = rgb_to_gray(mat)

            elif "Thresholding" in choice:
                result = thresholding(mat, 127)

            elif "Kontrast" in choice:
                result = contrast_adjustment(mat, alpha=1.5, beta=10)

            elif "Histogram Germe" in choice:
                result = histogram_stretching(mat if len(mat.shape) == 2 else rgb_to_gray(mat))

            elif "Histogram Eşitleme" in choice:
                gray = rgb_to_gray(mat) if len(mat.shape) == 3 else mat
                result = histogram_equalization(gray)

            elif "Histogram Görüntüle" in choice:
                gray = rgb_to_gray(mat) if len(mat.shape) == 3 else mat
                hist = calculate_histogram(gray)
                self.after(0, lambda: self._show_histogram(hist))
                self.after(0, lambda: self._finish_filter("Histogram gösterildi"))
                return

            elif "HSV" in choice:
                if len(mat.shape) == 3:
                    result = rgb_to_hsv(mat)
                else:
                    self.after(0, lambda: mb.showwarning("Uyarı", "HSV dönüşümü için renkli görüntü gerekli!"))
                    self.after(0, lambda: self._finish_filter("HSV: Renkli görüntü gerekli"))
                    return

            elif "YCbCr" in choice:
                if len(mat.shape) == 3:
                    result = rgb_to_ycbcr(mat)
                else:
                    self.after(0, lambda: mb.showwarning("Uyarı", "YCbCr dönüşümü için renkli görüntü gerekli!"))
                    self.after(0, lambda: self._finish_filter("YCbCr: Renkli görüntü gerekli"))
                    return

            elif "Döndür" in choice:
                result = rotate_crop_image(mat, 30)

            elif "Zoom" in choice:
                result = zoom_image(mat, scale=1.5)

            elif "Salt" in choice:
                result = add_salt_and_pepper_noise(mat, 0.05)

            elif "Mean" in choice:
                result = apply_mean_filter(mat if len(mat.shape) == 2 else rgb_to_gray(mat))

            elif "Median" in choice:
                result = apply_median_filter(mat if len(mat.shape) == 2 else rgb_to_gray(mat))

            elif "Motion" in choice:
                result = apply_motion_blur(mat if len(mat.shape) == 2 else rgb_to_gray(mat), 7)

            elif "Sobel" in choice:
                gray = rgb_to_gray(mat) if len(mat.shape) == 3 else mat
                result = sobel_edge_detection(gray)

            elif "Canny" in choice:
                gray = rgb_to_gray(mat) if len(mat.shape) == 3 else mat
                result = canny_edge_detection(gray)

            elif "Erosion" in choice:
                gray = rgb_to_gray(mat) if len(mat.shape) == 3 else mat
                binary = thresholding(gray)
                result = morph_erosion(binary, 3)

            elif "Dilation" in choice:
                gray = rgb_to_gray(mat) if len(mat.shape) == 3 else mat
                binary = thresholding(gray)
                result = morph_dilation(binary, 3)

            elif "Opening" in choice:
                gray = rgb_to_gray(mat) if len(mat.shape) == 3 else mat
                binary = thresholding(gray)
                result = morph_opening(binary, 3)

            elif "Closing" in choice:
                gray = rgb_to_gray(mat) if len(mat.shape) == 3 else mat
                binary = thresholding(gray)
                result = morph_closing(binary, 3)

            elif "Toplama" in choice or "Çıkarma" in choice or "AND" in choice or "OR" in choice or "XOR" in choice:
                if self.second_image_matrix is None:
                    self.after(0, lambda: mb.showwarning("Uyarı", "Önce '2. Resim Yükle' butonuyla\nikinci bir görüntü seçin!"))
                    self.after(0, lambda: self._finish_filter("2. resim gerekli"))
                    return
                op_map = {'Toplama': 'add', 'Çıkarma': 'subtract', 'AND': 'and', 'OR': 'or', 'XOR': 'xor'}
                op = 'add'
                for k, v in op_map.items():
                    if k in choice:
                        op = v
                        break
                result = image_arithmetic(mat, self.second_image_matrix, op)

            elif "PLAKA" in choice:
                plate_binary, chars, debug_info = segment_characters(mat)
                result = plate_binary
                # Debug görselleri ve karakterleri göster
                self.after(0, lambda d=debug_info, c=chars: self._show_lpr_result(d, c))

            if result is not None:
                if result.dtype != np.uint8:
                    result = np.clip(result, 0, 255).astype(np.uint8)
                self.current_image_matrix = result

            self.after(0, lambda: self._finish_filter(f"{choice} uygulandı ✓"))

        except Exception as e:
            err_msg = f"Filtre hatası: {e}\n{traceback.format_exc()}"
            print(err_msg)
            self.after(0, lambda: mb.showerror("İşlem Hatası", str(e)))
            self.after(0, lambda: self._finish_filter("Hata!"))

    def _finish_filter(self, status_msg):
        self.progress.stop()
        self.progress.set(0)
        self.btn_apply.configure(state="normal")
        self.display_images()
        self.set_status(status_msg)

    def _show_histogram(self, hist):
        try:
            plt.figure("Histogram")
            plt.clf()
            plt.bar(range(256), hist, color='gray', width=1)
            plt.title("Piksel Yoğunluk Histogramı")
            plt.xlabel("Piksel Değeri (0-255)")
            plt.ylabel("Frekans")
            plt.show(block=False)
        except Exception as e:
            print(f"Histogram gösterim hatası: {e}")

    def _show_chars(self, chars):
        try:
            max_show = min(len(chars), 15)
            fig, axes = plt.subplots(1, max_show, figsize=(12, 3))
            if max_show == 1:
                axes = [axes]
            for ax, ch_mat in zip(axes, chars[:max_show]):
                ax.imshow(ch_mat, cmap='gray')
                ax.axis('off')
            title = f"Segmentasyon: {len(chars)} karakter bulundu"
            if len(chars) > 15:
                title += f" (ilk 15 gösteriliyor)"
            plt.suptitle(title)
            plt.tight_layout()
            plt.show(block=False)
        except Exception as e:
            print(f"Karakter gösterim hatası: {e}")

    def _show_lpr_result(self, debug_info, chars):
        """
        Rapor.doc'taki LPR akışının TÜM adımlarını matplotlib ile gösterir:
        Satır 1: Gri Seviye | Mean Filter | Sobel Gx | Sobel Gy
        Satır 2: Sobel G (Kenar) | Edge Binary | Plaka Bölgesi | Plaka Binary
        Satır 3: Dikey Projeksiyon Histogramı (col_sums)
        Ayrı pencere: Segmente edilen karakterler
        """
        try:
            if not debug_info:
                print("LPR: debug_info boş, plaka bulunamadı.")
                return

            plt.close('all')

            # ─── Ana Figür: 3 satır ───
            fig, axes = plt.subplots(3, 4, figsize=(16, 10))
            fig.canvas.manager.set_window_title("Plaka Okuma - İşlem Adımları")

            # Satır 1: Ön işleme
            axes[0, 0].imshow(debug_info['gray'], cmap='gray')
            axes[0, 0].set_title("1. Gri Seviye Dönüşüm\n(Y=0.299R+0.587G+0.114B)", fontsize=8)
            axes[0, 0].axis('off')

            axes[0, 1].imshow(debug_info['filtered'], cmap='gray')
            axes[0, 1].set_title("2. Mean Filter\n(3×3 Konvolüsyon)", fontsize=8)
            axes[0, 1].axis('off')

            axes[0, 2].imshow(debug_info.get('sobel_gx', np.zeros((10,10))), cmap='gray')
            axes[0, 2].set_title("3a. Sobel Gx\n(Yatay Gradyan)", fontsize=8)
            axes[0, 2].axis('off')

            axes[0, 3].imshow(debug_info.get('sobel_gy', np.zeros((10,10))), cmap='gray')
            axes[0, 3].set_title("3b. Sobel Gy\n(Dikey Gradyan)", fontsize=8)
            axes[0, 3].axis('off')

            # Satır 2: Kenar + Plaka izolasyonu
            axes[1, 0].imshow(debug_info['edges'], cmap='gray')
            axes[1, 0].set_title("3c. G = √(Gx²+Gy²)\n(Kenar Büyüklüğü)", fontsize=8)
            axes[1, 0].axis('off')

            axes[1, 1].imshow(debug_info['edge_binary'], cmap='gray')
            axes[1, 1].set_title("4. Manuel Eşikleme\n(Binarization)", fontsize=8)
            axes[1, 1].axis('off')

            axes[1, 2].imshow(debug_info.get('plate_gray', np.zeros((10,10))), cmap='gray')
            thresh_val = debug_info.get('plate_thresh_value', '?')
            axes[1, 2].set_title(f"5. Plaka Bölgesi (Kırpma)\n(Projeksiyon ile bulundu)", fontsize=8)
            axes[1, 2].axis('off')

            axes[1, 3].imshow(debug_info.get('plate_binary', np.zeros((10,10))), cmap='gray')
            axes[1, 3].set_title(f"6. Plaka Binary\n(Otsu T={thresh_val})", fontsize=8)
            axes[1, 3].axis('off')

            # Satır 3: Dikey Projeksiyon Histogramı
            col_sums = debug_info.get('col_sums', np.array([]))
            
            # Sol 2 hücreyi birleştirerek geniş projeksiyon grafiği
            axes[2, 0].remove()
            axes[2, 1].remove()
            axes[2, 2].remove()
            axes[2, 3].remove()
            
            ax_proj = fig.add_subplot(3, 1, 3)
            if len(col_sums) > 0:
                ax_proj.bar(range(len(col_sums)), col_sums, color='steelblue', width=1)
                ax_proj.set_title("7. Dikey Projeksiyon (Sütun Bazlı Siyah Piksel Toplamı → Karakter Sınırları)", fontsize=9)
                ax_proj.set_xlabel("Sütun İndeksi (x)")
                ax_proj.set_ylabel("Siyah Piksel Sayısı")
                
                # Segmente edilen bölgeleri kırmızı ile işaretle
                if len(chars) > 0:
                    plate_h = debug_info.get('plate_binary', np.zeros((10,10))).shape[0]
                    threshold_sum = max(2, int(plate_h * 0.08))
                    ax_proj.axhline(y=threshold_sum, color='red', linestyle='--', label=f'Eşik={threshold_sum}', alpha=0.7)
                    ax_proj.legend(fontsize=8)
            else:
                ax_proj.text(0.5, 0.5, "Projeksiyon verisi yok", ha='center', va='center', fontsize=12)
                ax_proj.axis('off')

            fig.suptitle("PLAKA OKUMA (LPR) — İşlem Adımları", fontsize=14, fontweight='bold')
            fig.tight_layout()

            # ─── Karakter Figürü ───
            if len(chars) > 0:
                max_show = min(len(chars), 15)
                fig2, axes2 = plt.subplots(1, max_show, figsize=(max_show * 1.5, 3))
                fig2.canvas.manager.set_window_title("Bulunan Karakterler")
                
                if max_show == 1:
                    axes2 = [axes2]
                for i, ax in enumerate(axes2):
                    ax.imshow(chars[i], cmap='gray')
                    ax.set_title(f"Karakter #{i+1}", fontsize=9)
                    ax.axis('off')
                fig2.suptitle(f"8. Karakter Segmentasyonu — Toplam {len(chars)} karakter", fontsize=12, fontweight='bold')
                fig2.tight_layout()
            
            plt.show(block=False)

        except Exception as e:
            print(f"LPR gösterim hatası: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    app = App()
    app.mainloop()
