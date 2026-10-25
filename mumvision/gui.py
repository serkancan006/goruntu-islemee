"""Small Turkish desktop interface; all analysis stays local."""
import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .detector import annotate, detect
from .imaging import read_image, write_image
from .patterns import PATTERNS, TURKISH


class MumVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MumVision | Mum Formasyonu Tespiti")
        self.root.geometry("1160x790")
        self.root.minsize(900, 650)
        self.image = self.original = self.result = None
        self.photos, self.canvases = {}, {}
        self.drag_start = self.selection_rect = None
        self.display_scale, self.display_offset = 1.0, (0, 0)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=(12, 8))
        style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"))
        frame = ttk.Frame(root, padding=20)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="MumVision", style="Title.TLabel").pack(anchor="w")
        ttk.Label(frame, text="HSV renk ayrıştırma • Morfoloji • Mum geometrisi • 9 formasyon", font=("Segoe UI", 11)).pack(anchor="w", pady=(4, 14))
        toolbar = ttk.Frame(frame)
        toolbar.pack(fill="x", pady=(0, 10))
        for text, callback in (("Görüntü aç", self.choose), ("Örneği aç", self.example),
                               ("Kırpmayı sıfırla", self.reset), ("Sonucu kaydet", self.save), ("Deney raporu", self.report)):
            ttk.Button(toolbar, text=text, command=callback).pack(side="left", padx=(0, 8))
        ttk.Label(frame, text="Yalnız fiyat panelini kullanın. İlk sekmede fareyle alan seçerek menü, hacim ve indikatörleri kırpabilirsiniz.", wraplength=1050).pack(anchor="w", pady=(0, 10))
        notebook = ttk.Notebook(frame)
        notebook.pack(fill="both", expand=True)
        for name in ("Girdi / Alan seçimi", "Renk maskesi", "Tespit edilen formasyonlar"):
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=name)
            canvas = tk.Canvas(tab, background="#e8eaf0", highlightthickness=0)
            canvas.pack(fill="both", expand=True)
            self.canvases[name] = canvas
            canvas.bind("<Configure>", lambda event: self.redraw())
        original_canvas = self.canvases["Girdi / Alan seçimi"]
        original_canvas.bind("<ButtonPress-1>", self.crop_start)
        original_canvas.bind("<B1-Motion>", self.crop_drag)
        original_canvas.bind("<ButtonRelease-1>", self.crop_end)
        self.summary = tk.StringVar(value="Bir mum grafiği seçin veya hazır örneği açın.")
        ttk.Label(frame, textvariable=self.summary, font=("Segoe UI", 11, "bold"), wraplength=1050).pack(anchor="w", pady=(12, 5))
        self.details = tk.StringVar(value="Kırmızı/yeşil, içi dolu mumlar ve doğrusal fiyat ekseni desteklenir.")
        ttk.Label(frame, textvariable=self.details, wraplength=1050).pack(anchor="w")
        ttk.Label(frame, text="Sonuçlar geometrik formasyon adaylarıdır. Bu uygulama fiyat tahmini ölçmez.", foreground="#646b77").pack(anchor="w", pady=(10, 0))

    def choose(self):
        path = filedialog.askopenfilename(title="Mum grafiği seç", filetypes=[("Grafik görüntüleri", "*.png *.jpg *.jpeg *.bmp *.webp"), ("Tüm dosyalar", "*.*")])
        if path:
            try:
                self.load(path)
            except Exception as exc:
                messagebox.showerror("Görüntü açılamadı", str(exc))

    def example(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "outputs" / "demo_input.png"
        if not path.exists():
            candidates = sorted((root / "data" / "dataset" / "examples").glob("*.png"))
            if not candidates:
                messagebox.showinfo("Örnek", "Önce veri setini üretin: python -m mumvision build")
                return
            path = candidates[0]
        self.load(path)

    def load(self, path):
        self.original = read_image(path)
        self.image = self.original.copy()
        self.analyze()

    def reset(self):
        if self.original is not None:
            self.image = self.original.copy()
            self.analyze()

    def analyze(self):
        self.result = detect(self.image)
        counts = {name: sum(obj["name"] == name for obj in self.result.objects) for name in PATTERNS}
        last = [tr for tr, flag in zip(TURKISH, self.result.labels[-1])] if len(self.result.labels) else []
        self.summary.set(f"{len(self.result.ohlc)} mum · {len(self.result.objects)} formasyon adayı · Son mum: {', '.join(last) or 'Formasyon yok'}")
        parts = [f"{tr}: {counts[name]}" for name, tr in zip(PATTERNS, TURKISH) if counts[name]]
        self.details.set(" | ".join(parts) + ("\n" if parts and self.result.warnings else "") + " ".join(self.result.warnings))
        self.redraw()

    def redraw(self):
        if self.image is None or self.result is None:
            return
        mask = cv2.cvtColor(self.result.green_mask | self.result.red_mask, cv2.COLOR_GRAY2BGR)
        views = [self.image, mask, annotate(self.image, self.result)]
        for (name, canvas), view in zip(self.canvases.items(), views):
            width, height = max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 100)
            scale = min((width-24)/view.shape[1], (height-24)/view.shape[0], 2.5)
            scaled = Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)).resize((max(1, int(view.shape[1]*scale)), max(1, int(view.shape[0]*scale))), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(scaled)
            self.photos[name] = photo
            canvas.delete("all")
            x, y = (width-scaled.width)//2, (height-scaled.height)//2
            canvas.create_image(x, y, image=photo, anchor="nw")
            if name == "Girdi / Alan seçimi":
                self.display_scale, self.display_offset = scale, (x, y)

    def crop_start(self, event):
        if self.image is not None:
            self.drag_start = (event.x, event.y)

    def crop_drag(self, event):
        if self.drag_start is None:
            return
        canvas = self.canvases["Girdi / Alan seçimi"]
        if self.selection_rect is not None:
            canvas.delete(self.selection_rect)
        self.selection_rect = canvas.create_rectangle(*self.drag_start, event.x, event.y, outline="#0077cc", width=2)

    def crop_end(self, event):
        if self.drag_start is None:
            return
        x0, y0 = self.display_offset
        scale = self.display_scale
        first, second = self.drag_start, (event.x, event.y)
        self.drag_start, self.selection_rect = None, None
        xa, xb = sorted([int((first[0]-x0)/scale), int((second[0]-x0)/scale)])
        ya, yb = sorted([int((first[1]-y0)/scale), int((second[1]-y0)/scale)])
        xa, xb = max(0, xa), min(self.image.shape[1], xb)
        ya, yb = max(0, ya), min(self.image.shape[0], yb)
        if xb-xa >= 40 and yb-ya >= 40:
            self.image = self.image[ya:yb, xa:xb].copy()
            self.analyze()
        else:
            self.redraw()

    def save(self):
        if self.result is None:
            return
        destination = filedialog.askdirectory(title="Sonuçların kaydedileceği klasör")
        if destination:
            path = Path(destination)
            write_image(path / "mumvision_input.png", self.image)
            write_image(path / "mumvision_result.png", annotate(self.image, self.result))
            write_image(path / "mumvision_mask.png", self.result.green_mask | self.result.red_mask)
            (path / "mumvision_result.json").write_text(json.dumps(self.result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            messagebox.showinfo("Kaydedildi", f"Görüntü, maske ve JSON sonucu kaydedildi:\n{path}")

    def report(self):
        path = Path(__file__).resolve().parents[1] / "outputs" / "RAPOR.html"
        if path.exists():
            os.startfile(path)
        else:
            messagebox.showinfo("Rapor", "Önce değerlendirmeyi çalıştırın: python -m mumvision evaluate")


def main():
    root = tk.Tk()
    root.withdraw()
    MumVisionApp(root)
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()
