import numpy as np
import random as rng
import plotly.graph_objects as go


class RandomLine_2D:
    """ A randomized linear function \\
        f: R -> R ; f(x) = m*x + b
    """
    def __init__(self):
        self.m = rng.random()
        self.b = rng.random()

    def __call__(self,x):
        return x*self.m + self.b
# end


class RandomLine_3D:
    """ A randomized linear function \\
        f: R2 -> R ; f(x,y) = mx*x + my*y + b
    """
    def __init__(self):
        self.mx = rng.random()
        self.my = rng.random()
        self.b = rng.random()

    def __call__(self,x,y):
        return x*self.mx + y*self.my + self.b
# end


xy = np.linspace(-1, 1, 40)
fxy = RandomLine_3D()

noisy_x = [x + rng.uniform(-0.3, 0.3) for x in xy]
noisy_y = [y + rng.uniform(-0.3, 0.3) for y in xy]
noisy_z = [fxy(x,y) + rng.uniform(-0.3, 0.3) for (x,y) in zip(xy,xy)]

X = np.column_stack((np.ones((len(xy),1)), noisy_x, noisy_y))
a = np.linalg.inv(np.matmul(X.transpose(), X))
b = X.transpose().dot(noisy_z)
w = a.dot(b)
ols = lambda x,y: w[0] + w[1]*x + w[2]*y

trace1 = go.Scatter3d(
    x=xy, y=xy, z=ols(xy, xy),
    line=dict(width=10),
    name='best fit',
    mode='lines'
)

trace2 = go.Scatter3d(
    x=xy, y=xy, z=fxy(xy,xy),
    line=dict(width=10),
    name='original',
    mode='lines'
)

trace3 = go.Scatter3d(
    x=noisy_x, y=noisy_y, z=noisy_z,
    name='data',
    mode='markers'
)

layout = go.Layout(width=900, height=900)
fig = go.Figure(data=[trace1,trace2,trace3], layout=layout)
fig.show()