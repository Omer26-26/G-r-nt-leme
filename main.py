import customtkinter as ctk
from PIL import Image
import tkinter.filedialog as fd
import tkinter.messagebox as mb
import numpy as np
import threading
import traceback

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
        self.second_image_matrix = None

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
            img = self._read_image(filepath)
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
            img = self._read_image(filepath)
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
                mat = self.current_image_matrix
                if mat.dtype != np.uint8:
                    mat = np.clip(mat, 0, 255).astype(np.uint8)
                if len(mat.shape) == 2:
                    pil_img = Image.fromarray(mat, mode='L')
                else:
                    pil_img = Image.fromarray(mat)
                pil_img.save(filepath)
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
                pil_o = Image.fromarray(orig, mode='L') if len(orig.shape) == 2 else Image.fromarray(orig)
                pil_o.thumbnail(max_size, Image.Resampling.LANCZOS)
                self._ctk_img_orig = ctk.CTkImage(
                    light_image=pil_o, dark_image=pil_o,
                    size=(pil_o.size[0], pil_o.size[1])
                )
                self.panel_orig.configure(image=self._ctk_img_orig, text="")
            except Exception as e:
                print(f"Orijinal görüntü gösterim hatası: {e}")

        if self.current_image_matrix is not None:
            try:
                curr = self.current_image_matrix
                if curr.dtype != np.uint8:
                    curr = np.clip(curr, 0, 255).astype(np.uint8)
                pil_m = Image.fromarray(curr, mode='L') if len(curr.shape) == 2 else Image.fromarray(curr)
                pil_m.thumbnail(max_size, Image.Resampling.LANCZOS)
                self._ctk_img_mod = ctk.CTkImage(
                    light_image=pil_m, dark_image=pil_m,
                    size=(pil_m.size[0], pil_m.size[1])
                )
                self.panel_mod.configure(image=self._ctk_img_mod, text="")
            except Exception as e:
                print(f"İşlenmiş görüntü gösterim hatası: {e}")

    # ------------------------------------------------------------------
    def _read_image(self, filepath):
        """PIL kullanarak görüntüyü numpy array olarak okur."""
        try:
            pil_img = Image.open(filepath).convert("RGB")
            return np.array(pil_img)
        except Exception as e:
            print(f"Görüntü okuma hatası: {e}")
            return None

    # ------------------------------------------------------------------
    def apply_filter(self):
        if self.current_image_matrix is None:
            mb.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return

        choice = self.filter_option.get()
        self.set_status(f"İşleniyor: {choice}...")
        self.progress.start()
        self.btn_apply.configure(state="disabled")

        thread = threading.Thread(target=self._run_filter, args=(choice,), daemon=True)
        thread.start()

    def _run_filter(self, choice):
        try:
            mat = self.current_image_matrix.copy()
            result = None

            # TODO: Algoritmaları buraya yaz
            if "Gri Seviye" in choice:
                pass  # result = ...

            elif "Thresholding" in choice:
                pass  # result = ...

            elif "Kontrast" in choice:
                pass  # result = ...

            elif "Histogram Germe" in choice:
                pass  # result = ...

            elif "Histogram Eşitleme" in choice:
                pass  # result = ...

            elif "Histogram Görüntüle" in choice:
                self.after(0, lambda: self._finish_filter("Histogram (TODO)"))
                return

            elif "HSV" in choice:
                pass  # result = ...

            elif "YCbCr" in choice:
                pass  # result = ...

            elif "Döndür" in choice:
                pass  # result = ...

            elif "Zoom" in choice:
                pass  # result = ...

            elif "Salt" in choice:
                pass  # result = ...

            elif "Mean" in choice:
                pass  # result = ...

            elif "Median" in choice:
                pass  # result = ...

            elif "Motion" in choice:
                pass  # result = ...

            elif "Sobel" in choice:
                pass  # result = ...

            elif "Canny" in choice:
                pass  # result = ...

            elif "Erosion" in choice:
                pass  # result = ...

            elif "Dilation" in choice:
                pass  # result = ...

            elif "Opening" in choice:
                pass  # result = ...

            elif "Closing" in choice:
                pass  # result = ...

            elif "Toplama" in choice or "Çıkarma" in choice or "AND" in choice or "OR" in choice or "XOR" in choice:
                if self.second_image_matrix is None:
                    self.after(0, lambda: mb.showwarning("Uyarı", "Önce '2. Resim Yükle' butonuyla\nikinci bir görüntü seçin!"))
                    self.after(0, lambda: self._finish_filter("2. resim gerekli"))
                    return
                pass  # result = ...

            elif "PLAKA" in choice:
                pass  # result = ...

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

if __name__ == "__main__":
    app = App()
    app.mainloop()
