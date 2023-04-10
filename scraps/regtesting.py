from operator import itemgetter
from typing import Callable, Collection, Dict, Iterable, Sequence, Tuple, Any, List, Union
import re
import itertools


def tuple_to_ndim_dict(d:Dict[Tuple[Any,...],Any]):
  key = next(iter(d));
  if (len(key) == 1):
    return {k[0]:v for k,v in d.items()};
  else:
    keys = sorted(d.keys());
    out = {}
    for k,sub in itertools.groupby(keys,itemgetter(0)):
      out[k] = tuple_to_ndim_dict({k[1:]:d[k] for k in sub});
    return out

def attempt_apply(f:Callable,x):
    try:
        return f(x);
    except:
        # print(f,x);
        return x;

def regex_to_map(reg_str:str,targets:Sequence[str],keys:Union[Collection[str],Collection[int],None]=None,cast:Union[type,Callable,Iterable[type],Iterable[Callable],None]=None):
  regex = re.compile(reg_str);
  if keys is not None:
    ndims = len(keys);
  else:
    m = re.match(regex,targets[0])
    if m is None:
      raise Exception(f"regex did not match target {targets[0]}");
    ndims = len(m.groups()); #each group in order will be used as the dictionary key

  if cast and not isinstance(cast,Iterable): cast = [cast];
#   print(cast);
#   breakpoint();
  out:Dict[Tuple,str] = {};
  for target in targets:
    m = re.match(regex,target)
    if m is None:
      raise Exception(f"regex did not match target {target}");
    k = None
    if keys is not None:
      try:
        k = tuple(m.group(*keys));
      except IndexError:
        pass;
    if k is None:
      k = m.groups();
    if cast:
        k = tuple([attempt_apply(c,x) for c,x in zip(itertools.cycle(cast),k)])
    if len(k) != ndims:
      raise Exception(f"All regex matches must have the same number of extracted groups! Target {target} group count {len(k)} is inconsistent with {ndims}")
    out[k] = target

  #turn dict of tuples into n-dimensional dict
  return tuple_to_ndim_dict(out);

if __name__ == "__main__":
    from filegetter import askdirectory
    import os
    files = filter(lambda x: not x.endswith(".nd"),os.listdir(askdirectory()))
    regstr = r"p_s(\d+)_t(\d+).*"
    print(regex_to_map(regstr,list(files),cast=int));

