import csv
import tkinter as tk
from tkinter import filedialog
from typing import DefaultDict
from pathlib import Path
from itertools import zip_longest
import os
import subprocess


root = tk.Tk()
root.withdraw()

file_paths = [Path(s) for s in filedialog.askopenfilenames()];
out_file = file_paths[0];
for file_path in file_paths:
    data:dict[int,list[float]] = DefaultDict(lambda: []);

    with open(file_path,'r') as f:
        reader = csv.DictReader(f);
        for row in reader:
            if row['trackid'] != 'average':
                data[int(row['movie'])].append(float(row['FMI.y']));

    #output format is a raw csv file with 4 columns, each with the FMI data of one group of four movies
    c1 = data[1] + data[2] + data[3] + data[4];
    c2 = data[5] + data[6] + data[7] + data[8];
    c3 = data[9] + data[10] + data[11] + data[12];
    c4 = data[13] + data[14] + data[15] + data[16];

    out_file = Path('output_files')/file_path.name;
    if not os.path.exists(out_file.parent):
        os.makedirs(out_file.parent);
    
    if os.path.exists(out_file):
        os.remove(out_file);

    with open(out_file,'w') as f:
        writer = csv.writer(f,lineterminator='\n');
        for row in zip_longest(c1,c2,c3,c4,fillvalue=''):
            writer.writerow(row);

subprocess.Popen(fr'explorer /select,"{out_file}"')