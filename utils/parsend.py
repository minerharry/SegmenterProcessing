import itertools
from msilib.schema import Error
import random
import re
series_regex = "s([0-9]+)"
time_regex = "t([0-9]+)"
filename_regex = 'p[0-9]*_s([0-9]+)_t([0-9]+).*\.(TIF|TIFF|tif|tiff)';

##Parses .nd files from metamorph on the optotaxis microscope

def parseND(filePath):
    with open(filePath,'r') as f:
        lines = f.readlines();
    args = {};
    for line in lines:
        largs = line.split(", "); #line args lol
        if len(largs) == 0:
            assert largs[0] == "\"EndFile\"";
            break;
        args[largs[0].replace("\"","")] = largs[1].replace("\"","");
    return args;

def sorted_dir(paths:list[str]):
    def get_key(s:str):
        out = [];
        series = re.findall(series_regex,s);
        if series: 
            out.append(int(series[0]));
        else:
            print(s);
        time = re.findall(time_regex,s);
        if time:
            out.append(int(time[0]));
        else:
            print(s);
        return out;
    try:
        paths = filter(lambda s: s.endswith(".TIF"),paths);
        paths = sorted(paths,key=get_key);
    except Exception as e:
        print(e);
        print("hello my darling")
    return paths;

def stage_from_name(name:str):
    m = re.match(filename_regex,name);
    return m.group(1) if m else "-1";

def grouped_dir(paths:list[str]):
    out = [];
    for k,g in itertools.groupby(paths,stage_from_name):
        g = list(g)
        # print(g)
        if k == "-1": continue;
        out.append(sorted_dir(g));
    return out;

#groupby: itertools function that splits a list into sublists based on the value of a key function

    

# def getNDFileInfo(path):
#     args = parseND(path);
    
#     for arg,val in args.items():

print(sorted_dir([f's{i}_t{j}.TIF' for i in range(5) for j in range(30)]))