import sys
import pickle

print("wow")
a = sys.argv[1]
print(a)
with open(a,'rb') as f:
    q = pickle.load(f)

print(q['do_smoothing'])
    