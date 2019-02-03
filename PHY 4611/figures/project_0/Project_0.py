import sys
import math
import timeit
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10]

# Helper function for exercise 4.
def f1(x):
    return np.exp(-x/4)*np.sin(x)

# Helper function for exercise 4.
def f2(x):
    return np.exp(-x)*np.sin(x)
    
# Integer p is number of petals, integer w is 'pinch' factor; larger w = skinnier petals.
def rhodonea(p,w):
    x = np.linspace(0, 2*np.pi, 1000)
    y = [abs(np.sin(p*xx/2)**w) for xx in x]
    plt.polar(x,y)

# Print a matrix nicely.
def print_matrix(A):
    for row in A:
        print(" "*4, *row)
        
# My own (very slow) matrix multiplication routine, used for the FLOPS estimation exercise.        
def mat_mult(a, b):
    z = list(zip(*b))
    return [[sum(i*j for i,j in zip(k,m)) for m in z] for k in a]


#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#
#————————————————————————————————————————————————————— EXERCISES ——————————————————————————————————————————————————————#
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#

# Exercise 1 (Hello, world)
print("Greetings, Earthlings!")

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#

# Exercise 2 (simple matrix algebra).
A = [[1, 2], [3, 4]]
B = np.array(A)

print("A+A:")
print_matrix(np.array(A) + np.array(A))
print("B+B:")
print_matrix(B+B)
print("A+B:")
print_matrix(np.array(A)+B)
print("A-A:")
print_matrix(np.array(A) - np.array(A))
print("B-B:")
print_matrix(B-B)
print("2*A:")
print_matrix(2*np.array(A))
print("2*B:")
print_matrix(2*B)
print("A*A:")
print_matrix(np.array(A)@np.array(A))
print("B*B:")
print_matrix(np.matmul(B,B))
print("B.B:")
print_matrix(np.dot(B,B))
print("B^2:")
print_matrix(la.matrix_power(B,2))
print("B/B:")
print_matrix(B/B)

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#

# Exercise 3 (basic plots).
x = np.linspace(1,10,10)
y = x**2

# 3a
plt.plot(x,y)

# 3b
plt.plot(x,y,'+', mew = 2, markersize = 10) # mew = marker edge width

# 3c
plt.plot(x,y,'-',x,y,'+', mew = 2, markersize = 10)

# 3d
# The book is very ambiguous here. It's written in Python 2.7, but I could not find
# anything in the 2.7 documentation that describes what x(1:2:10) is supposed to do,
# so I assumed that it is simply an alternate syntax for array slicing. 
plt.plot(x,y,'r-',x[1:2:10],y[1:2:10],'bx', mew = 2, markersize = 20)

# 3e
plt.semilogy(x,y)

# 3f
plt.loglog(x,y)

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#

# Exercise 4 (reproduce plots from book).
x = np.linspace(0,20,1000)
y = [f1(i) for i in x]

# 4a
plt.plot(x,y,'-')

# 4b
plt.plot(x,y,'-',x,[np.exp(-i/4) for i in x],'--')

# 4c
x = np.arange(0.01,18.85,0.2)
plt.semilogy(x, [abs(np.exp(-i)*np.sin(i)) for i in x], '+')

# 4d
rhodonea(6, 2)

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#

# Exercise 5 (machine limits).
print(f"             max float:   {sys.float_info.max}")
print(f"         max expoonent:   {sys.float_info.max_exp}")
print(f"  max base 10 exponent:   {sys.float_info.max_10_exp}")
print()
print(f"             min float:   {sys.float_info.min}")
print(f"         min expoonent:   {sys.float_info.min_exp}")
print(f"  min base 10 exponent:   {sys.float_info.min_10_exp}")
print()
print(f"min \u03F5 in   (1+\u03F5)-1 = \u03F5:   {sys.float_info.epsilon}")
print("        max nxn matrix:   100000 x 100000")
print("    longest row vector:   10000000000 integer 1's")

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#

#Exercise 6 (FLOPS estimate)
# Initialize two nxn matrices of random digits.
n = 500
a = np.random.randint(-9, 9, (n, n))
b = np.random.randint(-9, 9, (n, n))

# Number of operations.
calcs = 2*n**3 - n**2

# Time mat_mult(a,b) over one trial.
mat_mult_timed = timeit.timeit('mat_mult(a, b)', globals=globals(), number=1)

# Rough estimation of FLOPS rounded to the nearest thousand.
print(f"FLOPS: {1000*round(calcs/(1000*mat_mult_timed))}")

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#

# Exercise 7 (interpf)
# TODO: try to curry this function.
def intrp(x, xx, yy):
    k = len(xx)
    if k != len(set(xx)):
        return "Cannot interpolate from duplicate x-values!"

    # Lagrange basis polynomial helper function.
    def lbp(j):
        p = 1
        for m in range(1,k):
            p *= (x - xx[(j+m)%k])/(xx[j] - xx[(j-m)%k])
        return p
        
    return sum(yy[j]*lbp(j) for j in range(k))


# f(x) = 1/3 (x-1)^5 - 1/4*x^4 - 3x^3 + 5(x-2)^2 - x
def f_test(x):
    return 1/3*(x-1)**5 - 1/4*x**4 - 3*x**3 + 5*(x-2)**2 - x

# Manually input some random x-values
x_ = [-3.24, -2.81, -1.31, 1.04, 2.78, 5.2]
# Evaluate x-values on the function we're trying to interpolate.
y_ = [f_test(x) for x in x_]

# Create a numpy linspace (xx) and evaluate (yy) so we can plot the interpolated curve.
xx = np.linspace(math.floor(min(x_)), math.ceil(max(x_)), 1000)
yy = [intrp(x, x_, y_) for x in xx]

# Create a numpy linspace (xf) and evalute (yf) on the original function so we can compare curves.
xf = np.linspace(math.floor(min(x_)), math.ceil(max(x_)), 100)
yf = [f_test(x) for x in xf]

# Show me the money!
plt.plot(xx, yy, 'b-', xf, yf, 'r--', x_, y_, 'g*')