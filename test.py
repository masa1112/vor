import pandas as pd

import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from collections import defaultdict
import itertools
import math
from PIL import Image

images = []
p = 1
df = pd.read_csv('pandastest.csv',sep=',')
lim = 0
while p < 101:
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

    #neiListにVNを格納(同じ者同士を同時に)
    for q in tri.vertices:
        for e, r in itertools.combinations(q, 2):
            neiList[e].add(r)
            neiList[r].add(e)

    print("fig %d " % (p))

    phi_sin = 0
    phi_cos = 0
    for key in sorted(neiList.keys()):
        #ソートし終わった各点のボロノイネイバーを表示
        enu = 0 #VNの数
        sin6 = 0
        cos6 = 0
        #print("%d:%s" % (key,','.join([str(h) for h in neiList[key]])))
        for h in neiList[key]:
            #keyに対してのVNの座標取得
            #print("%d(%f,%f)" % (h,x[h][0],x[h][1]))

            #key近傍の配向足し合わせる
            ang = math.atan2(x[h][1] - x[key][1],x[h][0] - x[key][0])
            sin6 += math.sin(ang*6)
            cos6 += math.cos(ang*6)
            enu += 1
        phi_sin += (1/enu) * sin6
        phi_cos += (1/enu) * cos6

    phi_len = (1 / int(len(x))) * ((phi_sin ** 2)+(phi_cos ** 2)) ** (1/2)
    #phi_6 = (1 / 6) * ((phi_sin ** 2) + (phi_cos ** 2)) ** (1 / 2)
    #print("phi_6 = %f" % (phi_6))
    print("phi_6 = %f" % (phi_len))

    vor = Voronoi(x)

    #軸の値域
    plt.xlim(0,650)
    plt.ylim(0,500)

    plt.figure(figsize=(6, 4),facecolor='white',)
    voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',line_width=1, line_alpha=0.3, point_size=1)

    #凡例表示
    '''
    for i, c in enumerate(x):
        plt.text(c[0], c[1], '#%d' % i, ha='center')
    '''
    #figにVoronoi diagramをsave
    plt.savefig('data/fig_{0}.png'.format(p))
    images.append(Image.open('data/fig_{0}.png'.format(p)))
    p += 1
#作成した全部をimages[]に入れてgif化
images[0].save('data/out.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
plt.show()

