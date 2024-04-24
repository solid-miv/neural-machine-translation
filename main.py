import tkinter as tk


def initialize_window():
    root = tk.Tk()

    root.title("Neural Machine Translator")

    root.geometry("500x400")  # Width x Height

    root.minsize(300, 200)  # Minimum width x Minimum height

    root.maxsize(800, 600)  # Maximum width x Maximum height

    return root


if __name__ == "__main__":
    root = initialize_window()

    root.mainloop()