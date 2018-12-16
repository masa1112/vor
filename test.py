import pandas as pd
import csv
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from PIL import Image
images = []
p = 0
for l in open('vor.csv').readlines():
    #data = l[:-1].split(',')
    data = [float(y.strip()) for y in l.split(',')]
    x = []

    for j in range(0,int(len(data)/2)):
        a = j * 2
        b = a + 1
        x.append([float(data[a]), float(data[b])])
    points = x
    vor = Voronoi(points)
    plt.xlim(0,1000)
    plt.ylim(0,40)
    plt.figure(figsize=(6, 4), facecolor='white')
    voronoi_plot_2d(vor)

    #plt.savefig('data/fig_{0}.png'.format(p))
    images.append(Image.open('data/fig_{0}.png'.format(p)))
    p += 1
images[0].save('data/out.gif', save_all=True, append_images=images[1:], duration=400, loop=0)
plt.show()