import mediapy as media
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()

path = filedialog.askopenfilename()
vid = media.read_video(path)
media.write_video(path,vid,fps=vid.metadata.fps*2);
