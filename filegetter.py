import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()

askdirectory = filedialog.askdirectory;
askopenfile = filedialog.askopenfile;
askopenfilename = filedialog.askopenfilename;
askopenfilenames = filedialog.askopenfilenames;
askopenfiles = filedialog.askopenfiles;
asksaveasfile = filedialog.asksaveasfile;
asksaveasfilename = filedialog.asksaveasfilename;