import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand

from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

# Fixing random state for reproducibility
np.random.seed(19680801)

x = np.random.rand(15)
y = np.random.rand(15)
z = np.random.rand(15)
c = np.random.rand(15)
s = np.random.rand(15) * 100


def onpick3(event):
    ind = event.ind
    print('onpick3 scatter:', ind, x[ind], y[ind])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
col = ax.scatter(x, y, z, s=s, c=c, picker=True)
fig.canvas.mpl_connect('pick_event', onpick3)
plt.show()
