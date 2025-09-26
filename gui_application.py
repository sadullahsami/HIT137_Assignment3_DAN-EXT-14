import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Dict, Any

# Stub models (real pipelines will come in a later commit)
from image_classification_model import ImageClassificationModel
from sentiment_analysis_model import SentimentAnalysisModel


class AIModelGUI:
    """GUI connected to model stubs: choose a model, provide input, process, and view results."""
    MODEL_IMAGE = "Image Classification"
    MODEL_TEXT = "Sentiment Analysis"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HIT137 – AI Model Integration")
        self.root.geometry("900x600")

        # UI state
        self.selected_model = tk.StringVar(value=self.MODEL_TEXT)

        self._build_ui()
        self._on_model_change()  # set initial control states

    # ------------------------- UI BUILD -------------------------
    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True)

        # Tabs
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
        self.model_combo.bind("<<ComboboxSelected>>", lambda e: self._on_model_change())

        self.browse_btn = ttk.Button(top, text="Browse Image…", command=self._on_browse)
        self.browse_btn.pack(side="left", padx=8)

        # Input
        input_frame = ttk.LabelFrame(self.tab_ai, text="Input", padding=12)
        input_frame.pack(fill="both", expand=False, padx=12, pady=8)

        self.input_text = tk.Text(input_frame, height=8, wrap="word")
        self.input_text.pack(fill="both", expand=True)

        # Actions
        action_frame = ttk.Frame(self.tab_ai, padding=12)
        action_frame.pack(fill="x")
        ttk.Button(action_frame, text="Process", command=self._on_process).pack(side="left")
        ttk.Button(action_frame, text="Clear", command=self._on_clear).pack(side="left", padx=8)

        # Output
        output_frame = ttk.LabelFrame(self.tab_ai, text="Output", padding=12)
        output_frame.pack(fill="both", expand=True, padx=12, pady=8)

        self.output_text = tk.Text(output_frame, height=12, wrap="word", state="disabled")
        self.output_text.pack(fill="both", expand=True)

        # --- MODEL INFORMATION TAB ---
        info_inner = ttk.Frame(self.tab_info, padding=12)
        info_inner.pack(fill="both", expand=True)

        ttk.Label(info_inner, text="Available Models", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.info_box = tk.Text(info_inner, height=16, wrap="word", state="disabled")
        self.info_box.pack(fill="both", expand=True, pady=(6, 0))

        self._populate_model_info()

        # --- OOP CONCEPTS TAB (placeholder content) ---
        oop_inner = ttk.Frame(self.tab_oop, padding=12)
        oop_inner.pack(fill="both", expand=True)
        oop_text = (
            "OOP Concepts used:\n"
            "• Abstraction via BaseModel (interface for all models)\n"
            "• Inheritance with concrete model classes\n"
            "• Polymorphism (same .process() for different inputs)\n"
            "• Encapsulation with private attrs (e.g., _model_name)\n"
            "• Decorators (@log_method_call, @validate_input)\n"
            "• Template/Factory patterns (will expand later)\n"
        )
        lbl = tk.Text(oop_inner, height=12, wrap="word")
        lbl.insert("1.0", oop_text)
        lbl.configure(state="disabled")
        lbl.pack(fill="both", expand=True)

    # ------------------------- EVENTS -------------------------
    def _on_model_change(self):
        model = self.selected_model.get()
        # Browse is useful for images; disable for text sentiment
        if model == self.MODEL_IMAGE:
            self.browse_btn.configure(state="normal")
        else:
            self.browse_btn.configure(state="disabled")

    def _on_browse(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            # Put the selected path into the input box
            self.input_text.delete("1.0", "end")
            self.input_text.insert("1.0", path)

    def _on_clear(self):
        self.input_text.delete("1.0", "end")
        self._set_output("")

    def _on_process(self):
        model_name = self.selected_model.get()
        user_input = self.input_text.get("1.0", "end").strip()

        if not user_input:
            messagebox.showwarning("Input required", "Please provide input (text or image path/URL).")
            return

        try:
            model = self._get_model(model_name)
            model.load_model()  # no-op in stubs
            result = model.process(user_input)
            self._render_result(result)
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed:\n{e}")

    # ------------------------- HELPERS -------------------------
    def _get_model(self, model_name: str):
        if model_name == self.MODEL_IMAGE:
            return ImageClassificationModel()
        return SentimentAnalysisModel()

    def _render_result(self, result: Any):
        """Accepts either list[dict] or any printable object from the stub."""
        if isinstance(result, list):
            lines: List[str] = []
            for i, item in enumerate(result, start=1):
                label = item.get("label", "N/A")
                score = item.get("score", "N/A")
                lines.append(f"{i}. {label} — score: {score}")
            self._set_output("\n".join(lines))
        else:
            self._set_output(str(result))

    def _set_output(self, text: str):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        if text:
            self.output_text.insert("1.0", text)
        self.output_text.configure(state="disabled")

    def _populate_model_info(self):
        info_items: List[Dict[str, Any]] = []
        for model in (ImageClassificationModel(), SentimentAnalysisModel()):
            info_items.append(model.get_model_info())

        lines = []
        for it in info_items:
            lines.append(f"- Name: {it.get('name')}\n  Type: {it.get('type')}\n")
        text = "Model registry:\n\n" + "\n".join(lines)

        self.info_box.configure(state="normal")
        self.info_box.delete("1.0", "end")
        self.info_box.insert("1.0", text)
        self.info_box.configure(state="disabled")

    # Public
    def run(self):
        self.root.mainloop()
