import pandas as pd
import csv
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

x=[[]]
for l in open('offw.csv').readlines():
    data = l[:-1].split(',')
#data = pd.read_csv('offw.csv', header=None, delim_whitespace=True, decimal=',')
#data = np.genfromtxt("offwb.txt", dtype = None, delimiter = "¥t")
for i in range(0,30):
    for j in range(0,100):
        #x += [[float(data[i][j*2]),float(data[i][j*2+1])]]

        #x[i][j] += [[data[i][j*2],data[i][j*2+1]]]
        x[j] += [[data[j * 2], data[j * 2 + 1]]]
    points = x
    vor = Voronoi(points)
    plt.figure(figsize=(6, 4), facecolor='white')
    ax = plt.subplot(aspect='equal')
    voronoi_plot_2d(vor)
#plt.show()
    plt.savefig('data/figure.png')