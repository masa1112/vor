import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, ConvexHull
from collections import defaultdict
import itertools
import math
from PIL import Image
import numpy as np
import matplotlib.cm as cm

def bond_order_parameter(x, neiList):
    phix = 0
    phi = np.zeros((x.shape[0], 2))
    for key in sorted(neiList.keys()):
        # ソートし終わった各点のボロノイネイバーを表示
        enu = 0  # VNの数
        sin6 = 0
        cos6 = 0
        ang = 0

        # ボンドオーダーパラメーター計算 angleの値域が違うかも
        for h in neiList[key]:
            # keyに対してのVNの座標取得
            # print("%d(%f,%f)" % (h,x[h][0],x[h][1]))

            # key近傍の配向足し合わせる
            ang = np.arctan2(x[h][1] - x[key][1], x[h][0] - x[key][0])
            sin6 += math.sin(ang * 6)
            cos6 += math.cos(ang * 6)
            enu += 1
        sin6 /= enu
        cos6 /= enu
        phi[key][0] = math.sqrt((sin6 ** 2) + (cos6 ** 2))
        if enu == 1:
            phi[key][0] == 0
            continue
        phix += math.sqrt((sin6 ** 2) + (cos6 ** 2))
    OP = phix / int(len(phi))
    return OP, phi

def are_histgram(p, x, vor):
    hist = np.zeros(x.shape[0])
    #hist =[]
    for index2, c in enumerate(x):
        try:
            ch = ConvexHull(vor.vertices[vor.regions[vor.point_region[index2]]])
        except:
            print("%d でpolygonが作れなかった" % p)
            p = p + 1
            # OP_F[p] = OP_F[p-1]
            continue
        hist[index2] = ch.volume * 0.3125 * 0.3125
        #hist[index2] = ch.volume

        #hist.append(ch.volume * 0.3125 * 0.3125)
    return hist

def hist_fig(p, hist, path):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_title('Area_hist_{0}'.format(p))
    edges = range(0, 200, 1)
    ax2.hist(hist, bins=edges, density=True)
    ax2 = plt.ylim(0, 0.5)
    ax2 = plt.xlabel("area ")
    ax2 = plt.ylabel("Frequency")
    fig2.savefig(path + 'hist/hist_{0}.png'.format(p))

def bop_fig(OP_F, path):
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.set_title('hexatic order parameter')
    ax3.plot(OP_F)
    ax3 = plt.ylim(0, 1.0)
    fig3.savefig(path + 'order_para.png')


def voronoi_fig(x, vor, hist, phi, path):
    fig1 = voronoi_plot_2d(vor, show_vertices=False, line_colors='blue', line_width=1, line_alpha=0.3, point_size=1)
    ax1 = fig1.add_subplot(1, 1, 1)
    for index2, dat in enumerate(hist):
        ax1.plot(x[index2][0], x[index2][1], color=cm.Blues(phi[index2][0] / 1), marker='o', markersize=5)
    ax1.set_title('Voronoi diagram_{0},BOP = %f'.format(p) % OP)
    ax1 = plt.xlim(0, 650)
    ax1 = plt.ylim(0, 500)
    # plt.imshow(phi,cmap="Blues")
    # plt.colorbar(fraction=0.046, pad=0.04)
    fig1.savefig(path + 'fig/fig_{0}.png'.format(p))


for sheet_num in range(1, 2):
    path = "data/iwasaki/sheet" + str(sheet_num) + "/"
    path_a = path + 'area' + str(sheet_num) + '.txt'
    path_w = path + 'bop' + str(sheet_num) + '.txt'
    s = 'New file'

    images = []
    histimage = []
    shape_num = 300
    # ylim = 400
    """
    if sheet_num == 2 :
        shape_num = 101
        #ylim = 2000
    if sheet_num == 3 :
        shape_num = 101
        #ylim = 2000
    if sheet_num == 4 :
        shape_num = 201
        #ylim = 1000
    """

    # OP_F = np.zeros(shape_num-1)
    OP_F = []
    OP_txt = []
    area_txt =[]
    p = 1
    try:
        df = pd.read_csv('csv/iwasaki/sheet' + str(sheet_num) + '.csv', sep=',')
    except:
        break
    area_fig =[]
    while p < shape_num:
        dft = df[df['Slice'] == p]
        x = np.array([dft["XM"] , dft["YM"] ]).T

        try:
            tri = Delaunay(x)
        except:
            break
        vor = Voronoi(x)
        neiList = defaultdict(set)
        for q in tri.vertices:
            for e, r in itertools.combinations(q, 2):
                neiList[e].add(r)
                neiList[r].add(e)

        OP, phi = bond_order_parameter(x, neiList)
        OP_txt.append(str(p) + '\t' + str(OP) + '\n')
        OP_F.append(OP)

        hist = are_histgram(p, x, vor)
        hist = hist[hist < 1000]
        histmean = np.mean(hist)
        histvar = np.var(hist)
        area_fig.append(histmean)

        area_txt.append(str(p) + '\t' + str(histmean) + '\t' +str(histvar) + '\n')

        hist_fig(p, hist, path)
        histimage.append(Image.open(path + 'hist/hist_{0}.png'.format(p)))

        voronoi_fig(x, vor, hist, phi, path)
        images.append(Image.open(path + 'fig/fig_{0}.png'.format(p)))

        print("fig %d " % (p))

        p += 1

    with open(path_a, mode='w') as f:
        f.writelines(area_txt)
        f.close
    with open(path_w, mode='w') as f:
        f.writelines(OP_txt)
        f.close

    images[0].save(path + 'out.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
    histimage[0].save(path + 'histout.gif', save_all=True, append_images=histimage[1:], duration=100, loop=0)
    bop_fig(OP_F, path)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.set_title('area')
    ax3.plot(area_fig)
    #ax3 = plt.ylim(0, 1.0)
    fig3.savefig(path + 'area.png')
    """
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(1, 1, 1)
    ax4.set_title('BOP_hist')
    #edges = range(0, 1, 0.1)
    ax4.hist(phi[:,0], bins=100,range=(0,1))
    ax4 = plt.xlabel("BOP")
    ax4 = plt.ylabel("number")
    ax4 = plt.xlim(0,1)
    fig4.savefig(path + 'BOP_hist.png')
    """