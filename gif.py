import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay,ConvexHull
from collections import defaultdict
import itertools
import math
from PIL import Image
import numpy as np

p = 1
images = []
histimage =[]
while p < 212:
    images.append(Image.open('tanami/colloid/5V/data/fig_{0}.png'.format(p)))
    histimage.append(Image.open('tanami/colloid/5V/hist/hist_{0}.png'.format(p)))
    p += 1
images[0].save('tanami/colloid/5V/data/out.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
histimage[0].save('tanami/colloid/5V/hist/histout.gif', save_all=True, append_images=histimage[1:], duration=100, loop=0)