# coding: utf-8
'''
功能跟： create_list.sh 一样，只不用python 重写一遍
'''

import os
# 数据集的根目录
root_folder = r'/homec/wyj/dataset/VOCdevkit'
dataset_folder = 'VOC2012'
# 输出的txt放在哪个目录
output_dir = "/homec/wyj/dataset/VOCdevkit/RefineDet"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_file = os.path.join(output_dir,'trainval.txt')

trainval_path = '/ImageSets/Main/trainval.txt'

ANNO_PREFIX = "Annotations/"
IMAGE_PREFIX  = "JPEGImages/"


def create_list():
    full_path = os.path.join(root_folder, dataset_folder + trainval_path)
    lines = []
    with open(full_path, 'r') as f:
        lines = f.readlines()
    if len(lines) == 0:
        print("error to read file!")
        return
    out = []
    for line in lines:
        o = line.strip()
        o = IMAGE_PREFIX + o + ".jpg " + ANNO_PREFIX + o + ".xml\n"
        out += [o]
    print("all num is: {}".format(len(out)))

    with open(output_file, 'w') as f:
        for o in out:
            f.write(o)



if __name__ == '__main__':
    create_list()
    print('done')






