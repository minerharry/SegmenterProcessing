class SafeDict(dict[str,str]):
    def __missing__(self, key:str):
        return '{' + key + '}'