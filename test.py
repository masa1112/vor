import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay,ConvexHull
from collections import defaultdict
import itertools
import math
from PIL import Image
import numpy as np
import matplotlib.cm as cm

path = "data/iwasaki/sheet3/"
images = []
histimage = []

OP_F = []
p = 1
df = pd.read_csv('csv/iwasaki/sheet3.csv',sep=',')
lim = 0

while p < 200:

    #これいじるのだるいからどうにかできる
    x = []
    dft = df[df['time'] == p]
    for i in range(lim,lim + int(len(dft.index))):
        dfx = dft[['xx']]
        dfy = dft[['yy']]
        x.append([float(dfx.loc[i]), float(dfy.loc[i])])
    lim += int(len(dft.index))

    '''
for l in open('vor.csv').readlines():
    #読み込んだのをdataに
    data = [float(y.strip()) for y in l.split(',')]

    x = []

    for j in range(0,int(len(data)/2)):
        a = j * 2
        b = a + 1
        #x[]に2次元配列でx,yを入れる->[[x1,y1],[x2,y2],....]
        x.append([float(data[a]), float(data[b])])
    '''
    #points = x
    tri = Delaunay(x)
    neiList = defaultdict(set)
    arr_x = np.array(x)

    #neiListにVNを格納(同じ者同士を同時に)
    for q in tri.vertices:
        for e, r in itertools.combinations(q, 2):
            neiList[e].add(r)
            neiList[r].add(e)

    print("fig %d " % (p))

    phix = 0
    phi = np.zeros((arr_x.shape[0], 2))
    phi_hist = []
    for key in sorted(neiList.keys()):
        #ソートし終わった各点のボロノイネイバーを表示
        enu = 0 #VNの数
        sin6 = 0
        cos6 = 0
        ang = 0

        #ボンドオーダーパラメーター計算 angleの値域が違うかも
        for h in neiList[key]:
            #keyに対してのVNの座標取得
            #print("%d(%f,%f)" % (h,x[h][0],x[h][1]))

            #key近傍の配向足し合わせる
            ang = np.arctan2(x[h][1] - x[key][1],x[h][0] - x[key][0])
            sin6 += math.sin(ang*6)
            cos6 += math.cos(ang*6)
            enu += 1
        sin6 /= enu
        cos6 /= enu
        phi_ang = np.arctan2(cos6,sin6)
        phi[key][0] = math.sqrt((sin6 ** 2)+(cos6 ** 2))
        phi_hist.append(math.sqrt((sin6 ** 2)+(cos6 ** 2)))
        phix += math.sqrt((sin6 ** 2)+(cos6 ** 2))
    OP = 0
    OP = phix / int(len(phi))
    OP_F.append(OP)

    print("phi_6 = %f" % (OP))

    vor = Voronoi(x)

    fig1 = voronoi_plot_2d(vor, show_vertices=False, line_colors='blue',line_width=1, line_alpha=0.3, point_size=1)

    '''
    for region, c in zip([r for r in vor.regions if -1 not in r and r], ['yellow', 'pink']):
        ax.fill(vor.vertices[region][:, 0],
                vor.vertices[region][:, 1],
                color=c)
    voronoiPlot2Dのax が使えないから色指定できない気がする
    '''
    #凡例表示

    hist =[]
    for u, c in enumerate(x):
        #plt.text(c[0], c[1], '#%d' % i, ha='center')
        ch = ConvexHull(vor.vertices[vor.regions[vor.point_region[u]]])
        #print('volume:', ch.volume)
        hist.append(ch.volume)

    ax1 = fig1.add_subplot(1, 1, 1)

    local_area = np.zeros(arr_x.shape[0])
    for index, point in enumerate(x):
        local_area[index] = ConvexHull(np.vstack(
            (vor.vertices[[s for s in vor.regions[vor.point_region[index]] if s >= 0]], vor.points[index]))).area

    for index2, dat in enumerate(local_area):
        ax1.plot(arr_x[index2][0], arr_x[index2][1], color=cm.Blues(phi[index2][0] / 1), marker='o', markersize=5)
        #ax1.triplot(arr_x[:, 0], arr_x[:, 1], tri.simplices, color="b")


    #figにVoronoi diagramをsave
    ax1 = fig1.add_subplot(1, 1, 1)

    ax1.set_title('Voronoi diagram_{0},BOP = %f'.format(p) % OP)
    ax1 = plt.xlim(0,650)
    ax1 = plt.ylim(0,500)
    #plt.imshow(phi,cmap="Blues")
    #plt.colorbar(fraction=0.046, pad=0.04)
    fig1.savefig(path + 'fig/fig_{0}.png'.format(p))
    images.append(Image.open(path + 'fig/fig_{0}.png'.format(p)))


    #histgramのgif作る

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_title('Volume_hist_{0}'.format(p))
    edges = range(0, 1000, 5)
    ax2.hist(hist, bins=edges)
    ax2 = plt.xlabel("Volume")
    ax2 = plt.ylabel("number")
    ax2 = plt.ylim(0, 50)
    fig2.savefig(path + 'hist/hist_{0}.png'.format(p))
    histimage.append(Image.open(path + 'hist/hist_{0}.png'.format(p)))
    #fig2.show()



    p += 1
#作成した全部をimages[]に入れてgif化
images[0].save(path + 'out.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
histimage[0].save(path + 'histout.gif', save_all=True, append_images=histimage[1:], duration=100, loop=0)
fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)
ax3.set_title('bond order parameter')
ax3.plot(OP_F)
fig3.savefig(path + 'order_para.png')

fig4 = plt.figure()
ax4 = fig4.add_subplot(1, 1, 1)
ax4.set_title('BOP_hist')
#edges = range(0, 1, 0.1)
ax4.hist(phi_hist, bins=100,range=(0,1))
ax4 = plt.xlabel("BOP")
ax4 = plt.ylabel("number")
ax4 = plt.xlim(0,1)
fig4.savefig(path + 'BOP_hist.png')
#fig3.show()
#plt.show()

