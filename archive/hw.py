import numpy as np
import random as rand
import plotly.graph_objects as go


def rng(a=1.0, b=-1.0): return rand.uniform(a,b)


class Line2D:
    """ A randomized linear function \\
        f: R -> R ; f(x) = m*x + b
    """
    def __init__(self, m=rng(), b=rng()):
        self.m = m
        self.b = b

    def __call__(self, x):
        return x*self.m + self.b
# end


class Line3D:
    """ A randomized linear function \\
        f: R2 -> R ; f(x,y) = mx*x + my*y + b
    """
    def __init__(self, mx=rng(), my=rng(), b=rng()):
        self.mx = mx
        self.my = my
        self.b = b

    def __call__(self, x, y):
        return x*self.mx + y*self.my + self.b
# end


def reg(xx, yy) -> 'function':
    """ Linear regression using analytic minimized loss \\
        Returns a regression line in Rm as a lambda \\
        f: R(nxm) -> g: Rm -> R
    """
    X = np.column_stack((np.ones((len(xx),1)), xx))
    A = np.linalg.inv(np.matmul(X.transpose(), X))
    b = X.transpose().dot(yy)
    w = A.dot(b)
    
    return lambda x: np.column_stack((np.ones((len(x),1)), x)).dot(w)
# end