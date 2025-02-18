


from typing import Any, Iterator


def next_default(itera:Iterator,default:Any):
    try:
        return next(itera)
    except StopIteration:
        return default


def zip_dict(d:dict[str,list[Any]],fillvalue:Any=""):
    """made for writing a dict of lists with csvwriter"""
    maxlen = max(len(x) for x in d.values());
    iters = {k:iter(d[k]) for k in d}
    for _ in range(maxlen):
        yield {k:next_default(iters[k],fillvalue) for k in d};