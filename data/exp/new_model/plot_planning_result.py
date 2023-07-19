import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
import os
import sys
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TimesNewRoman = FontProperties(fname='/home/jiayy/Times New Roman/times new roman.ttf')
# font_manager.fontManager.addfont('/home/jiayy/Times New Roman/Times New Roman.ttf')

FONT_SIZE = 18

plt.rcParams['pdf.fonttype'] = 42
item_name_list = ['Time', 'Length']
y_label = ['Planning Time', 'Track Length']
title_list = ['Planning Time (s)', 'Path Length (m)']
method = ['CA3P','RRTc']
ylim_list = [#[0, 800],\
             [0, 150], \
             #[0, 0.006], \
             [0, 2.3]]
# sns.set(font=TimesNewRoman.get_name())
sns.set_theme(style="white",font='Times New Roman',font_scale=1.2)
for idx, item_name in enumerate(item_name_list):
    exp_data = pd.read_csv('./'+item_name+'.csv')
    exp_data['Method'] = exp_data.Method.map(lambda x: 'CA3P' if x =='R3T' else 'RRTc')
    # exp_data = pd.read_csv(item_name+'.csv')
    # import pdb; pdb.set_trace()
    splt = sns.catplot(
        kind = "box",
        x = "Scene",
        y = item_name_list[idx],
        hue = "Method",
        data = exp_data,
        legend=False,
        legend_out=False
    )
    splt.set(ylabel=None)

    plt.ylim(ylim_list[idx])
    ax = plt.gca()
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    plt.grid(which='major', axis='both')
    plt.title(title_list[idx], loc='center', fontsize=FONT_SIZE)
    plt.xlabel("Scene", fontsize=FONT_SIZE)

    #设置坐标刻度字体大小
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    
    plt.tick_params(left=False)

    if y_label[idx] == 'Planning Time':
        plt.xlim([-0.5, 4.5])
        plt.hlines(y=130, xmin=-0.5, xmax=4.5, linestyles="--")
        plt.text(x=1.58, y=133, s="failure", fontsize=FONT_SIZE)
        plt.legend(bbox_to_anchor=(0.55, 0.6), fontsize=FONT_SIZE)
    elif y_label[idx] == 'Track Length':
        plt.xlim([-0.5, 4.5])
        plt.hlines(y=2.0, xmin=-0.5, xmax=4.5, linestyles="--")
        plt.text(x=1.58, y=2.04, s="failure", fontsize=FONT_SIZE)

    # plt.show()
    # ax.set_aspect(1.0)
    plt.savefig(y_label[idx]+'.pdf', bbox_inches='tight')
