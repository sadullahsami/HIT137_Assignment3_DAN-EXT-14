import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Dict, Any
import threading

from image_classification_model import ImageClassificationModel
from sentiment_analysis_model import SentimentAnalysisModel
from base_model import friendly_error_message
from utils import is_url, load_config, save_config, clamp_int

class AIModelGUI:
    """GUI: select model, input data, threaded processing, settings with persistence."""
    MODEL_IMAGE = "Image Classification"
    MODEL_TEXT = "Sentiment Analysis"

    SAMPLE_IMAGE_URL = "https://huggingface.co/datasets/mishig/sample_images/resolve/main/dog.jpg"
    SAMPLE_TEXT = "I absolutely love this product! It works better than I expected."

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HIT137 – AI Model Integration")
        self.root.geometry("920x700")

        # Load persisted config (or defaults)
        self.config = load_config()

        # Menus
        self._build_menu()

        # UI state
        self.selected_model = tk.StringVar(value=self.MODEL_TEXT)

        self._build_ui()
        self._on_model_change()
        self._update_hint()

    # ------------------------- MENUS -------------------------
    def _build_menu(self):
        menubar = tk.Menu(self.root)

        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Preferences…", command=self._open_settings)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Quick Guide", command=self._show_help)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _show_help(self):
        msg = (
            "Quick Guide\n\n"
            "1) Choose a model from the dropdown.\n"
            "2) For Image Classification: paste an image URL or click ‘Browse Image…’\n"
            "   For Sentiment Analysis: type/paste any text.\n"
            "3) Click ‘Load Sample’ for an instant demo.\n"
            "4) Click ‘Process’ and wait while the model runs.\n"
            "5) Use ‘Copy Output’ to put results on your clipboard.\n"
            "6) Settings → Preferences… to configure Top-K (image) and Max Text Length."
        )
        messagebox.showinfo("Quick Guide", msg)

    def _show_about(self):
        messagebox.showinfo(
            "About",
            "HIT137 – AI Model Integration GUI\n"
            "ViT for images, RoBERTa for sentiment.\n"
            "Includes persistent Settings (Top-K, Max Text Length)."
        )

    def _open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Preferences")
        win.transient(self.root)
        win.grab_set()

        frm = ttk.Frame(win, padding=16)
        frm.pack(fill="both", expand=True)

        # Top-K (image)
        ttk.Label(frm, text="Image Top-K predictions (1–10):").grid(row=0, column=0, sticky="w", pady=(0,6))
        topk_var = tk.IntVar(value=int(self.config.get("image_top_k", 5)))
        topk_spin = tk.Spinbox(frm, from_=1, to=10, textvariable=topk_var, width=6)
        topk_spin.grid(row=0, column=1, sticky="w", pady=(0,6), padx=(8,0))

        # Max text length
        ttk.Label(frm, text="Sentiment Max Text Length (32–2048):").grid(row=1, column=0, sticky="w", pady=(0,6))
        maxlen_var = tk.IntVar(value=int(self.config.get("text_max_len", 256)))
        maxlen_spin = tk.Spinbox(frm, from_=32, to=2048, increment=32, textvariable=maxlen_var, width=6)
        maxlen_spin.grid(row=1, column=1, sticky="w", pady=(0,6), padx=(8,0))

        # Buttons
        btns = ttk.Frame(frm)
        btns.grid(row=2, column=0, columnspan=2, sticky="e", pady=(12,0))
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right")
        def on_save():
            new_topk = clamp_int(topk_var.get(), 1, 10)
            new_maxlen = clamp_int(maxlen_var.get(), 32, 2048)
            self.config["image_top_k"] = new_topk
            self.config["text_max_len"] = new_maxlen
            save_config(self.config)
            messagebox.showinfo("Preferences", "Settings saved.")
            win.destroy()
        ttk.Button(btns, text="Save", command=on_save).pack(side="right", padx=(0,8))

        for i in range(2):
            frm.columnconfigure(i, weight=1)

    # ------------------------- UI BUILD -------------------------
    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True)

        self.tab_ai = ttk.Frame(nb)
        self.tab_info = ttk.Frame(nb)
        self.tab_oop = ttk.Frame(nb)

        nb.add(self.tab_ai, text="AI Models")
        nb.add(self.tab_info, text="Model Information")
        nb.add(self.tab_oop, text="OOP Concepts")

        # --- AI MODELS TAB ---
        top = ttk.Frame(self.tab_ai, padding=12)
        top.pack(fill="x")

        ttk.Label(top, text="Model:").pack(side="left")
        self.model_combo = ttk.Combobox(
            top,
            textvariable=self.selected_model,
            values=[self.MODEL_IMAGE, self.MODEL_TEXT],
            state="readonly",
            width=28,
        )
        self.model_combo.pack(side="left", padx=8)
        self.model_combo.bind("<<ComboboxSelected>>", lambda e: (self._on_model_change(), self._update_hint()))

        self.browse_btn = ttk.Button(top, text="Browse Image…", command=self._on_browse)
        self.browse_btn.pack(side="left", padx=8)

        # Input
        input_frame = ttk.LabelFrame(self.tab_ai, text="Input", padding=12)
        input_frame.pack(fill="both", expand=False, padx=12, pady=8)

        self.input_text = tk.Text(input_frame, height=8, wrap="word")
        self.input_text.pack(fill="both", expand=True)
        self.input_text.bind("<KeyRelease>", lambda e: self._update_hint())

        # Detection hint
        self.hint_lbl = ttk.Label(input_frame, text="", foreground="")
        self.hint_lbl.pack(anchor="w", pady=(6, 0))

        # Actions
        action_frame = ttk.Frame(self.tab_ai, padding=12)
        action_frame.pack(fill="x")
        self.process_btn = ttk.Button(action_frame, text="Process", command=self._on_process)
        self.process_btn.pack(side="left")
        self.clear_btn = ttk.Button(action_frame, text="Clear", command=self._on_clear)
        self.clear_btn.pack(side="left", padx=8)
        self.sample_btn = ttk.Button(action_frame, text="Load Sample", command=self._on_load_sample)
        self.sample_btn.pack(side="left", padx=0)
        self.copy_btn = ttk.Button(action_frame, text="Copy Output", command=self._on_copy_output)
        self.copy_btn.pack(side="left", padx=8)

        # Status + Progress
        status_frame = ttk.Frame(self.tab_ai, padding=(12, 0, 12, 12))
        status_frame.pack(fill="x")
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate")
        self.status_lbl = ttk.Label(status_frame, text="Idle.")
        self.status_lbl.pack(side="left")
        self.progress.pack(side="right", fill="x", expand=True, padx=(12, 0))

        # Output
        output_frame = ttk.LabelFrame(self.tab_ai, text="Output", padding=12)
        output_frame.pack(fill="both", expand=True, padx=12, pady=8)

        self.output_text = tk.Text(output_frame, height=14, wrap="word", state="disabled")
        self.output_text.pack(fill="both", expand=True)

        # --- MODEL INFORMATION TAB ---
        info_inner = ttk.Frame(self.tab_info, padding=12)
        info_inner.pack(fill="both", expand=True)

        header = ttk.Frame(info_inner)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="Available Models", font=("TkDefaultFont", 11, "bold")).pack(side="left")
        ttk.Button(header, text="Refresh", command=self._populate_model_info).pack(side="right")

        self.info_box = tk.Text(info_inner, height=18, wrap="word", state="disabled")
        self.info_box.pack(fill="both", expand=True)

        self._populate_model_info()

        # --- OOP CONCEPTS TAB ---
        oop_inner = ttk.Frame(self.tab_oop, padding=12)
        oop_inner.pack(fill="both", expand=True)
        oop_text = (
            "OOP Concepts used:\n"
            "• Abstraction: BaseModel defines the interface\n"
            "• Inheritance: models derive from BaseModel\n"
            "• Polymorphism: same .process() with different behavior\n"
            "• Encapsulation: private attrs (e.g., _model_name, _pipeline)\n"
            "• Decorators: validation & logging\n"
            "• Settings persistence via utils.load_config/save_config"
        )
        lbl = tk.Text(oop_inner, height=12, wrap="word")
        lbl.insert("1.0", oop_text)
        lbl.configure(state="disabled")
        lbl.pack(fill="both", expand=True)

    # ------------------------- EVENTS -------------------------
    def _on_model_change(self):
        model = self.selected_model.get()
        self.browse_btn.configure(state="normal" if model == self.MODEL_IMAGE else "disabled")

    def _on_browse(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.input_text.delete("1.0", "end")
            self.input_text.insert("1.0", path)
            self._update_hint()

    def _on_clear(self):
        self.input_text.delete("1.0", "end")
        self._set_output("")
        self._update_hint()

    def _on_load_sample(self):
        sample = self.SAMPLE_IMAGE_URL if self.selected_model.get() == self.MODEL_IMAGE else self.SAMPLE_TEXT
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", sample)
        self._update_hint()

    def _on_copy_output(self):
        text = self.output_text.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Copy Output", "Nothing to copy.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        messagebox.showinfo("Copy Output", "Output copied to clipboard.")

    def _on_process(self):
        model_name = self.selected_model.get()
        user_input = self.input_text.get("1.0", "end").strip()
        if not user_input:
            if model_name == self.MODEL_TEXT:
                messagebox.showwarning("Input required", "Please enter some text to analyse.")
            else:
                messagebox.showwarning("Input required", "Please provide an image URL or choose a local file.")
            return

        self._start_busy("Loading model & processing…")
        t = threading.Thread(target=self._do_process, args=(model_name, user_input), daemon=True)
        t.start()

    # ------------------------- BACKGROUND WORK -------------------------
    def _do_process(self, model_name: str, user_input: str):
        try:
            model = self._get_model(model_name)

            # Apply settings to the model instance
            if isinstance(model, ImageClassificationModel):
                model._top_k = clamp_int(self.config.get("image_top_k", 5), 1, 10)
            else:
                model._max_len = clamp_int(self.config.get("text_max_len", 256), 32, 2048)

            model.load_model()
            result = model.process(user_input)
            self.root.after(0, lambda: self._render_result(result))
        except Exception as e:
            msg = friendly_error_message(e)
            self.root.after(0, lambda: messagebox.showerror("Processing failed", msg))
        finally:
            self.root.after(0, self._stop_busy)

    # ------------------------- HELPERS -------------------------
    def _get_model(self, model_name: str):
        if model_name == self.MODEL_IMAGE:
            return ImageClassificationModel()
        return SentimentAnalysisModel()

    def _render_result(self, result: Any):
        if isinstance(result, list):
            self._set_output(self._format_predictions(result))
        else:
            self._set_output(str(result))

    def _format_predictions(self, preds: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for i, item in enumerate(preds, start=1):
            label = str(item.get("label", "N/A"))
            try:
                score = float(item.get("score", 0))
            except Exception:
                score = 0.0
            bar = self._bar(score, width=24)
            pct = f"{score * 100:.1f}%"
            lines.append(f"{i:>2}. {label}\n    {bar} {pct}")
        return "\n".join(lines)

    def _bar(self, score: float, width: int = 24) -> str:
        if score < 0: score = 0.0
        if score > 1: score = 1.0
        filled = int(round(score * width))
        return "█" * filled + "·" * (width - filled)

    def _set_output(self, text: str):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        if text:
            self.output_text.insert("1.0", text)
        self.output_text.configure(state="disabled")

    def _populate_model_info(self):
        items: List[Dict[str, Any]] = []
        for model in (ImageClassificationModel(), SentimentAnalysisModel()):
            info = model.get_model_info()
            items.append(info)

        lines = []
        for it in items:
            lines.append(
                "- Name: {name}\n"
                "  Type: {type}\n"
                "  Task: {task}\n"
                "  Provider: {provider}\n"
                "  Notes: {notes}\n".format(
                    name=it.get("name", "N/A"),
                    type=it.get("type", "N/A"),
                    task=it.get("task", "N/A"),
                    provider=it.get("provider", "N/A"),
                    notes=it.get("notes", "—"),
                )
            )

        text = "Model registry:\n\n" + "\n".join(lines)
        self.info_box.configure(state="normal")
        self.info_box.delete("1.0", "end")
        self.info_box.insert("1.0", text)
        self.info_box.configure(state="disabled")

    def _update_hint(self):
        model = self.selected_model.get()
        payload = self.input_text.get("1.0", "end").strip()
        if model == self.MODEL_IMAGE:
            if not payload:
                msg = "Tip: Paste an image URL (http/https) or click ‘Browse Image…’ to pick a local file."
            elif is_url(payload):
                msg = "Detected: Image URL"
            elif os.path.exists(payload):
                msg = "Detected: Local file path"
            else:
                msg = "Input not recognized as URL/path. Provide a valid image URL or file path."
        else:
            if not payload:
                msg = "Tip: Type or paste any text, then click ‘Process’."
            else:
                msg = f"Text length: {len(payload)} characters (max {self.config.get('text_max_len', 256)})"
        self.hint_lbl.configure(text=msg)

    # Busy UI helpers
    def _start_busy(self, msg: str):
        self.status_lbl.configure(text=msg)
        self.process_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")
        self.browse_btn.configure(state="disabled")
        self.sample_btn.configure(state="disabled")
        self.copy_btn.configure(state="disabled")
        self.progress.start(12)

    def _stop_busy(self):
        self.progress.stop()
        self.status_lbl.configure(text="Idle.")
        self.process_btn.configure(state="normal")
        self.clear_btn.configure(state="normal")
        self.sample_btn.configure(state="normal")
        self.copy_btn.configure(state="normal")
        self._on_model_change()

    def run(self):
        self.root.mainloop()
