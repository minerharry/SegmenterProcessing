import os
import subprocess
import tempfile
from typing import Any, Literal
import warnings

from bs4 import BeautifulSoup, NavigableString

#list of jars to include by default in the class math. Usually either bioformats_package.jar location,
# or the locations of both formats-gpl.jar and bio-formats-tools.jar
bf_folder = r"C:\Program Files\bftools"
bf_jars = [bf_folder,rf"{bf_folder}\bioformats_package.jar"]

bf_aux_files = [r"utils\logback"]; #make sure logging output is nice, specifically neded for formatlist. Needs to be folder for godforsaken reason

class CommandException(Exception):
    def __init__(self,command,message):
        self.command = command
        self.message = message
        super().__init__(f"Error executing command: {self.command}\n" + self.message);
    pass;

class BF_Commands:
    SHOWINF = "loci.formats.tools.ImageInfo";
    IJVIEW = "loci.plugins.in.Importer"; #imagej view requires imagej jar as well
    BFCONVERT = "loci.formats.tools.ImageConverter";
    FORMATLIST = "loci.formats.tools.PrintFormatTable"
    XMLINDENT = "loci.formats.tools.XMLIndent";
    XMLVALID = "loci.formats.tools.XMLValidate";
    TIFFCOMMENT = "loci.formats.tools.TiffComment";
    DOMAINLIST = "loci.formats.tools.PrintDomains";
    MKFAKE = "loci.formats.tools.ImageFaker";



## the camelcase all caps variables are all to match the environment variable names and conventions in the commandline bftools
def call_bioformats(mainclass:str,
                    *command_args:str,
                    exclude_bf_jars:bool=False,
                    BF_MAX_MEM:str="512m",
                    NO_UPDATE_CHECK:bool|None=None,
                    PROXY_HOST:str|None=None,
                    PROXY_PORT:str|None=None,
                    BF_PROFILE:bool|None=None,
                    BF_PROFILE_DEPTH:int=30,
                    jvm_args:list[str]=[],
                    classpath:str|list[str]=[],
                    decode:str|bool="utf-8",
                    block_stdin:bool=True):
    """Call bioformats program. Format based off of OME's bftools command line syntax. 
    jvm_args are passed to the jvm separated by a space.
    any jars in classpath will be included running the program.
    *args are passed to the program separated by a space.
    In total the java command will look (roughly) like this:
    `java [jvm_args] -cp [class1;class2;...;<bf_jars classes if not exclude_bf_jars>] mainclass [command_args]

    """


    ## make flags, assemble command

    flags:dict[str,Any] = {}; #jvm flags
    BF_PROG = mainclass; #which program to run
    
    flags["Dbioformats_can_do_upgrade_check"] = "false" if bool(NO_UPDATE_CHECK) else "true";

    if (bool(BF_PROFILE)):
        # BF_PROFILE_DEPTH = int(BF_PROFILE_DEPTH);
        flags["agentlib:hprof"] = ",".join([
            "cpu=samples",
            "depth=" + str(BF_PROFILE_DEPTH),
            "file=" + BF_PROG + ".hprof"
        ]); ##no idea what this means
    
    if PROXY_HOST is not None:
        flags["Dhttp.proxyHost"] = PROXY_HOST;
    if PROXY_PORT is not None:
        flags["Dhttp.proxyPort"] = PROXY_PORT;
    
    if isinstance(classpath,str):
        classpath = [classpath];

    if not exclude_bf_jars:
        classpath += bf_jars;
        classpath += bf_aux_files;

    classpath = classpath.copy();
    jvm_args = jvm_args.copy(); #clear reference to default argument
    jvm_args.append(f"-Xmx{BF_MAX_MEM}");

    jvm_args += [f"-{k}={v}" for k,v in flags.items()];
    jvm_args += ["-cp",";".join(classpath)];

    class_args = list(command_args)# + [f"-{k} " + " ".join(v) for k,v in command_kwargs];

    command = ["java",*jvm_args,mainclass,*class_args];

    print(command)

    ## run command
    print("Calling bftools...")
    try:
        out = subprocess.run(command,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,stdin=subprocess.PIPE if block_stdin else None); #redirect stdin iff prevent user from entering
    except FileNotFoundError as e:
        raise Exception("java executable not found, make sure it's on your PATH environemnt variable");
    except subprocess.CalledProcessError as e:
        
        print(e.stdout.decode())
        raise CommandException(command,e.stderr.decode()) from e;
    print("bftools called")

    stdout = out.stdout

    if decode == True: decode = "utf-8"
    if decode:
        return stdout.decode(decode);
    else:
        return stdout;

#oh boy lots of flags
def showinf(file:str,nopix:bool|None=None,nocore:bool|None=None,nometa:bool|None=None,thumbs:bool|None=None,minmax:bool|None=None,
            merge:bool|None=None,nogroup:bool|None=None,stitch:bool|None=None,separate:bool|None=None,expand:bool|None=None,
            omexml:bool|None=None,normalize:bool|None=None,fast:bool|None=None,debug:bool|None=None,preload:bool|None=None,
            autoscale:bool|None=None,novalid:bool|None=None,omexml_only:bool|None=None,no_sas:bool|None=None,no_upgrade:bool|None=None,
            noflat:bool|None=None,trace:bool|None=None,ascii:bool|None=None,nousedfiles:bool|None=None,
            range:tuple[int,int]|None=None,series:int|None=None,resolution:int|None=None,swap:str|None=None, shuffle:str|None=None,
            map:str|None=None,crop:tuple[int,int,int,int]|None=None,format:str|None=None,xmlversion:str|None=None,
            xmlspaces:int|None=None,
            cache:bool|None=None,cache_dir:str|None=None,options:dict[str,str]={},fill:int|None=None,
            **kwargs):
    """BY DEFAULT, THIS COMMAND OPENS A BLOCKING JAVA WINDOW TO BE INTERACTED WITH! use the nopix option to disable the window. 
    See full documentation here https://bio-formats.readthedocs.io/en/stable/users/comlinetools/display.html"""
    command_kwargs = locals().copy(); #capture all kwargs
    del command_kwargs["file"];
    del command_kwargs["options"];
    del command_kwargs["kwargs"];
    decode = kwargs.get("decode",False) 
    kwargs["decode"]=False; #ok so idk why but showinf decoding is fucked up so...

    commands = [file]

    for k,v in command_kwargs.items():
        if v is None or v == False: continue;
        k = k.replace("_","-");
        commands.append(f"-{k}");
        match k:
            case "xmlversion" | "xmlspaces" | "series" | "resolution" \
                | "swap" | "shuffle" | "map" | "format" | "cache-dir" | "fill": #single-input flags
                commands.append(str(v))
                break;
            case "range":
                commands += [str(v[0]),str(v[1])]
                break;
            case "crop":
                commands.append(",".join(v))
                break;

    for k,v in options.items():
        commands += ["-option",k,v];

    mainclass = BF_Commands.SHOWINF;
    res:bytes = call_bioformats(mainclass,*commands,**kwargs);

    if decode:
        if decode == True: decode = "utf-8"
        try:
            return res.decode(decode)
        except UnicodeDecodeError as f:
            warnings.warn("Unicode decoding resulted in an error: " + str(f))
            return res.decode(decode,errors="replace")

    else:
        return res



def get_omexml_metadata(file:str,
                        xmlversion:str|None=None,xmlspaces:int|None=None,cache:bool|None=None,cache_dir:str|None=None,
                        no_sas:bool|None=None,
                        options:dict[str,str]={},**kwargs)->str:
    extra_args = locals().copy();
    del extra_args["file"];
    del extra_args["kwargs"];
    extra_args.update(kwargs);
    return showinf(file,nopix=True,omexml_only=True,decode=True,**extra_args);

#oh boy lots of flags
def bfconvert(infile:str,outfile:str,
              debug:bool|None=None,stitch:bool|None=None,separate:bool|None=None,merge:bool|None=None,
              nogroup:bool|None=None,nolookup:bool|None=None,autoscale:bool|None=None,no_sas:bool|None=None,
              novalid:bool|None=None,validate:bool|None=None,padded:bool|None=None,noflat:bool|None=None,
              precompressed:bool|None=None,expand:bool|None=None,no_sequential:bool|None=None,
              ##dual flags: if set to false *or* true, should do the proper flag
              overwrite:bool=False,
              bigtiff:bool|None=None,
              ##flags with arguments
              map:str|None=None,crop:tuple[int,int,int,int]|None=None,compression:str|None=None,
              channel:int|None=None,z:int|None=None,timepoint:int|None=None,series:int|None=None,
              range:tuple[int,int]|None=None,tilex:int|None=None,tiley:int|None=None,pyramid_scale:int|None=None,
              pyramid_resolution:int|None=None,swap:str|None=None,fill:int|None=None,
              extra_metadata:str|None=None,cache:bool|None=None,cache_dir:str|None=None,
              options:dict[str,str]={},
              **kwargs):
    """See full documentation here https://bio-formats.readthedocs.io/en/v7.2.0/users/comlinetools/conversion.html"""
    command_kwargs = locals().copy(); #capture all kwargs
    del command_kwargs["infile"];
    del command_kwargs["outfile"];
    del command_kwargs["options"];
    del command_kwargs["kwargs"];

    commands = [infile,outfile]

    for k,v in command_kwargs.items():
        if v is None: continue;
        if v == False:
            if (k == "overwrite" or k == "bigtiff"):
                k = "no" + k;
            else:
                continue;
        k = k.replace("_","-");
        commands.append(f"-{k}");
        match k:
            case "map" | "compression" | "cache-dir" | "channel" | "z" | "timepoiint" \
                "series" | "tilex" | "tiley" | "pyramid-scale" | "pyramid-resolution" \
                "swap" | "fill" | "extra-metadta" : #single-input flags
                commands.append(str(v))
                break;
            case "range":
                assert(isinstance(v,tuple))
                commands += [str(v[0]),str(v[1])]
                break;
            case "crop":
                assert(isinstance(v,tuple))
                commands.append(",".join(v))
                break;

    for k,v in options.items():
        commands += ["-option",k,v];

    mainclass = BF_Commands.BFCONVERT;
    return call_bioformats(mainclass,*commands,**kwargs);

#TODO: why does this have a bunch of java nonsense happening? It's like the debug level is too high
def formatlist(format:Literal["txt","html","xml"]="txt", **kwargs):
    """See full documentation here https://bio-formats.readthedocs.io/en/v7.2.0/users/comlinetools/formatlist.html"""
    mainclass = BF_Commands.FORMATLIST;
    return call_bioformats(mainclass,f"-{format}",**kwargs);

def formatlist_dict(**kwargs):
    """Returns dict of {Plugin:{"ext":["jpg","png",...],"read":True/False,"write":True/False,"writemultipage":True/False}}"""
    soup = BeautifulSoup(formatlist("xml",**kwargs),"lxml-xml");

    wordmap = {"reading":"read","writing":"write","writing multiple pages":"writemultipage"};

    res:dict[str,dict[Literal["ext","read","write","writemultipage"],list[str]|bool]] = {}
    for c in soup.find("response").children:
        if (c is None or isinstance(c,NavigableString)): continue
        plugin = c["name"];
        data:dict[str,list[str]|bool] = {"read":False,"write":False,"writemultipage":False};
        for s in c.find_all_next("tag",{"name":"support"}):
            data[wordmap[s["value"]]] = True;
        data["ext"] = c.find("tag",{"name":"extensions"})["value"].split("|");
        res[plugin] = data;
    return res


def supported_formats(category:Literal['read','write','writemultipage','all']|list[Literal['read','write','writemultipage']]='all',**kwargs):
    """Returns list of supported formats (without the dot), e.g. ["png","tif",...] etc"""
    format_dict = formatlist_dict();
    formats:set[str] = set();

    if category == "all":
        category = ["read","write","writemultipage"];
    if isinstance(category,str):
        category = [category];

    for format,data in format_dict.items():
        if all([data[c] for c in category]):
            formats.update(data["ext"]);

    return list(formats)

def xmlindent(files:str|list[str]|None,valid:bool=False,block_stdin:bool=True,**kwargs):
    """NOTE: if files is empty or None, will use stdin for filename; this only works if the block_stdin kwarg is set to False.
    See full documentation here https://bio-formats.readthedocs.io/en/v7.2.0/users/comlinetools/xmlindent.html"""
    if files is None:
        files = [];
    if isinstance(files,str):
        files = [files];

    if (len(files) == 0 and block_stdin):
        raise ValueError("You must set block_stdin to false to enter xml file path using stdin");

    command = files;
    if valid:
        command.append("-valid");

    mainclass = BF_Commands.XMLINDENT;
    return call_bioformats(mainclass,*command,block_stdin=block_stdin,**kwargs);

def xmlvalid(files:str|list[str]|None,block_stdin:bool=True,**kwargs):
    """See full documentation here https://bio-formats.readthedocs.io/en/v7.2.0/users/comlinetools/xml-validation.html"""
    if files is None:
        files = [];
    if isinstance(files,str):
        files = [files];

    if (len(files) == 0 and block_stdin):
        raise ValueError("You must set block_stdin to false to enter xml file path using stdin");

    command = files;

    mainclass = BF_Commands.XMLVALID;
    return call_bioformats(mainclass,*command,block_stdin=block_stdin,**kwargs);


def tiffcomment(files:str|list[str],set_comment:str|None=None,edit:bool=False,block_stdin:bool=True,**kwargs):
    """NOTE: by default, will block the "-" option to read from stdin. To enable this, set block_stdin to False.
    See full documentation here https://bio-formats.readthedocs.io/en/stable/users/comlinetools/edit.html"""


    if set_comment == "-" and block_stdin == True:
        raise ValueError("You must set block_stdin to false to enter comments using stdin");

    if isinstance(files,str):
        files = [files]
    commands = files;
    if (set_comment is not None):
        commands += ["-set",set_comment];
    if (edit):
        commands += ["-edit"];


    mainclass = BF_Commands.TIFFCOMMENT;
    return call_bioformats(mainclass,*commands,block_stdin=block_stdin,**kwargs);

def domainlist(**kwargs):
    """See full documentation here https://bio-formats.readthedocs.io/en/v7.2.0/users/comlinetools/domainlist.html"""
    
    mainclass = BF_Commands.DOMAINLIST;
    return call_bioformats(mainclass,**kwargs);

def domainlist_dict(**kwargs)->dict[str,list[str]]:
    """Calls :py:func:`domainlist` and parses output into a dict of lists of strings {domain:[*formats]} """
    if "decode" in kwargs:
        del kwargs["decode"];
    domains:str = domainlist(decode="utf-8",**kwargs);
    res:dict[str,list[str]] = {};
    current_domain = None;
    for line in domains.split("\n"):
        line = line.strip();
        if line == "":
            continue;
        if line.endswith(":"):
            assert not line.startswith(" ");
            current_domain = line.rstrip(":");
            res[current_domain] = [];
        else:
            if current_domain is None:
                continue;
            # assert current_domain is not None,line
            res[current_domain].append(line);
    
    return res;

def mkfake(outfile:str,plates:int|None=None,runs:int|None=None,rows:int|None=None,columns:int|None=None,
           fields:int|None=None,debug:bool|None=None,**kwargs):
    """See full documentation here https://bio-formats.readthedocs.io/en/v7.2.0/users/comlinetools/mkfake.html"""
    command_kwargs = locals().copy(); #capture all kwargs
    del command_kwargs["outfile"];
    del command_kwargs["kwargs"];

    commands = [outfile]

    for k,v in command_kwargs.items():
        if v is None or v == False: continue;
        k = k.replace("_","-");
        commands.append(f"-{k}");

    mainclass = BF_Commands.MKFAKE;
    return call_bioformats(mainclass,*commands,**kwargs);
    
