import pandas as pd
import csv
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from collections import defaultdict
import itertools

from PIL import Image
images = []
p = 0
s = 0
for l in open('vor.csv').readlines():
    #data = l[:-1].split(',')
    data = [float(y.strip()) for y in l.split(',')]
    x = []

    for j in range(0,int(len(data)/2)):
        a = j * 2
        b = a + 1
        x.append([float(data[a]), float(data[b])])
    points = x
    tri = Delaunay(points)
    neiList = defaultdict(set)
    for q in tri.vertices:
        for e, r in itertools.combinations(q, 2):
            neiList[e].add(r)
            neiList[r].add(e)
    print("fig ( %d )" % (s))
    s += 1
    for key in sorted(neiList.keys()):
        print("%d:%s" % (key, ','.join([str(h) for h in neiList[key]])))
    vor = Voronoi(points)
    #plt.xlim(0,1000)
    #plt.ylim(0,40)
    plt.figure(figsize=(6, 4), facecolor='white')
    voronoi_plot_2d(vor)
    #vr = vor.region
    for i, c in enumerate(x):
        plt.text(c[0], c[1], '#%d' % i, ha='center')

    plt.savefig('data/fig_{0}.png'.format(p))
    images.append(Image.open('data/fig_{0}.png'.format(p)))
    p += 1
images[0].save('data/out.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
plt.show()