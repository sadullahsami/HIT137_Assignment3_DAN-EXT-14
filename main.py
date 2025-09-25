try:
import tkinter as tk
except Exception as e:
print("Tkinter not available:", e)
raise

def main():
    root = tk.Tk()
    root.title("HIT137 – Prototype")
    tk.Label(root, text="HIT137 Assignment 3 (Prototype)").pack(padx=24, pady=24)
    root.mainloop()

if __name__ == "__main__":
    main()