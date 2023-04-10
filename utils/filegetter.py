import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()

askdirectory = filedialog.askdirectory;
askopenfilehandle = filedialog.askopenfile;
askopenfilename = filedialog.askopenfilename;
askopenfilenames = filedialog.askopenfilenames;
askopenfiles = filedialog.askopenfiles;
asksaveasfilehandle = filedialog.asksaveasfile;
asksaveasfilename = filedialog.asksaveasfilename;