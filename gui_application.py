import tkinter as tk
from tkinter import ttk

class AIModelGUI:
    """Minimal GUI class with tabbed layout placeholders."""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HIT137 – AI Model Integration")
        self.root.geometry("900x600")
        self._build_ui()

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

        ttk.Label(self.tab_ai, text="Select a model and provide input.").pack(pady=20)
        ttk.Label(self.tab_info, text="Model details will appear here.").pack(pady=20)
        ttk.Label(self.tab_oop, text="OOP explanations will appear here.").pack(pady=20)

    def run(self):
        self.root.mainloop()
