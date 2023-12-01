import functools
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from shapely.geometry import box,Polygon,Point
import matplotlib.pyplot as plt
from tqdm import tqdm
# from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint
import cvxpy as cp

def ackley(a,b,c,x,y):
    d = 2
  
    sum1 = x**2 + y**2
    sum2 = np.cos(c*x) + np.cos(c*y)

    term1 = -a * np.exp(-b*np.sqrt(sum1/d))
    term2 = -np.exp(sum2/d)

    z = term1 + term2 + a + np.exp(1)
    return z

star = Polygon([(1,1),(4,0),(1,-1),(0,-4),(-1,-1),(-4,0),(-1,1),(0,4)])

shape = star
# for x in tqdm(np.linspace(-4,4,80)):
#     for y in tqdm(np.linspace(-4,4,80),leave=False):
#         if star.contains(Point(x,y)):
#             plt.scatter(x,y,color='green')
#         else:
#             plt.scatter(x,y,color='red');

# plt.show()

# func = lambda center: ackley(20,0.2,2*np.pi,center[0]+4,center[1]-3) #global minimum at (-4,3)
func = lambda center: -1/(np.power(center[0]+4,2) + np.power(center[1]-3,2))

fig, ax = plt.subplots()

# Make data.
X = np.linspace(-5, 5, 40)
Y = np.linspace(-5, 5, 40)
X, Y = np.meshgrid(X, Y)
# X,Y = 
# R = np.sqrt(X**2 + Y**2)
Z = func((X,Y))
print(Z.shape)
print(X.shape)

start = [0,0]

pos = cp.Variable(2)

constraint = [(lambda pos: star.contains(Point(pos)))]

objective = cp.Minimize(func)

problem = cp.Problem(objective,constraint)
print(problem.solve)

# constraint = NonlinearConstraint(lambda p: int(star.contains(Point(p))),1,1)
# res = minimize(func,start,bounds=[(-5,5),]*2,constraints=[])
# print(res)
# res2 = minimize(func,start,bounds=[(-5,5),]*2,constraints=[constraint])
# print(res2)

# # print(X,Y)
# # plt.scatter(np.reshape(X,(-1,)),np.reshape(Y,(-1,)),)
# plt.scatter(X,Y,c=Z)

# plt.scatter(*res.x,color='red')
# plt.scatter(*res2.x,color='blue')
# # print(res2.x)
# print(star.exterior.coords)
# plt.plot(*np.array(star.exterior.coords).T)
# plt.show()