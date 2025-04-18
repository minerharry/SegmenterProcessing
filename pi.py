import itertools
def chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield itertools.chain([first], itertools.islice(iterator, size - 1))

def make_pi():
    q, r, t, k, m, x = 1, 0, 1, 1, 3, 3
    for j in range(1000):
        if 4 * q + r - t < m * t:
            yield m
            q, r, t, k, m, x = 10*q, 10*(r-m*t), t, k, (10*(3*q+r))//t - 10*m, x
        else:
            q, r, t, k, m, x = q*k, (2*q+r)*x, t*x, k+1, (q*(7*k+2)+r*x)//(t*x), x+2

def make_pi_real():
    it = make_pi()
    yield next(it)
    yield "."
    yield from it

# my_array = []


for i in chunks(make_pi_real(),4):
    print("".join(map(str,i)),end=" ")
    

# my_array = my_array[:1] + ['.'] + my_array[1:]
# big_string = "".join(my_array)
# print("here is a big string:\n %s" % big_string)