from collections import UserDict
import itertools
import re
from typing import Dict, List, Tuple, TypeVar
from libraries.parse_moviefolder import filename_regex
series_regex = "s([0-9]+)"
time_regex = "t([0-9]+)"

##Parses .nd files from metamorph on the optotaxis microscope
T = TypeVar("T")
null = object()
class NDData(dict[str,str|list[str]]):
    def get(self,val:str,default:T|None=None):
        if val in self:
            if isinstance(self[val],str):
                return self[val]
            else:
                raise ValueError(f"Duplicate entry detected for key {val}, use __getitem__ or getEntry instead")
        else:
            if default:
                return default
            else:
                raise KeyError(val)
            
    def getEntry(self,val:str,default:T=null):
        if val in self:
            return self[val]
        else:
            if default is not null:
                return default
            else:
                raise KeyError(val)

def parseND(filePath)->NDData:
    with open(filePath,'r') as f:
        lines = f.readlines();
    args = NDData();
    for line in lines:
        largs = line.rstrip("\n").split(", "); #line args lol
        if largs[0] == '':
          continue;
        if len(largs) == 1 or largs[1] == '':
            if largs[0].startswith("\"EndFile\""):
              break;
            continue;
        key = largs[0].replace("\"","")
        val = ", ".join(larg.replace("\"","") for larg in largs[1:]);
        if key in args:
            ##DUPLICATE ROW! This happens sometimes. result is a list of str for each instance
            if isinstance(args[key],list):
                args[key].append(val)
            else:
                args[key] = [args[key],val]
        else:
            args[key] = val
    return args;

def StageDict(filePath):
    result:Dict[int,str] = {}
    data = parseND(filePath);
    for i in itertools.count(1):
        try:
            result[i] = (data[f"Stage{i}"])
        except KeyError:
            # print(f"Stage{i}")
            break;
    return result;
    
    

def sorted_dir(paths:List[str]):
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
    return m.group(2) if m else "-1";

def grouped_dir(paths:List[str]):
    out = [];
    for k,g in itertools.groupby(paths,stage_from_name):
        g = list(g)
        # print(g)
        if k == "-1": continue;
        out.append(sorted_dir(g));
    return out;


##takes a stage dict and groups stages with the same nonnumeric prefix together (good for the metamorph stage position naming scheme)
def group_stage_basenames(stage_dict:Dict[int,str]):
    invmap = {v:k for k,v in stage_dict.items()};
    order = sorted(invmap.keys());
    print(order)
    grouped = itertools.groupby([(k,invmap[k]) for k in order],key=lambda t: re.split("\\d",t[0])[0])
    groups:Dict[str,List[Tuple[str,int]]] = {}
    for k1,k2 in grouped:
        res = groups[k1] = []
        for a1,a2 in k2:
            res.append((a1,a2))
    return groups


#groupby: itertools function that splits a list into sublists based on the value of a key function

