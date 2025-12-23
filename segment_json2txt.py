# -*- coding: utf-8 -*-
from tqdm import tqdm
import shutil
import random
import os
import argparse
from collections import Counter
import yaml
import json
 
 

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
 
 
def convert_label_json(json_dir, save_dir, classes):
    json_paths = os.listdir(json_dir)
    classes = classes.split(',')
    mkdir(save_dir)
 
    for json_path in tqdm(json_paths):
        # for json_path in json_paths:
        path = os.path.join(json_dir, json_path)
        with open(path, 'r') as load_f:
            json_dict = json.load(load_f)
        h, w = json_dict['imageHeight'], json_dict['imageWidth']
 
        # save txt path
        txt_path = os.path.join(save_dir, json_path.replace('json', 'txt'))
        txt_file = open(txt_path, 'w')
 
        for shape_dict in json_dict['shapes']:
            label = shape_dict['label']
            label_index = classes.index(label)
            points = shape_dict['points']
 
            points_nor_list = []
 
            for point in points:
                points_nor_list.append(point[0] / w)
                points_nor_list.append(point[1] / h)
 
            points_nor_list = list(map(lambda x: str(x), points_nor_list))
            points_nor_str = ' '.join(points_nor_list)
 
            label_str = str(label_index) + ' ' + points_nor_str + '\n'
            txt_file.writelines(label_str)
 
 
def get_classes(json_dir):
    '''
    统计路径下 JSON 文件里的各类别标签数量
    '''
    names = []
    json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')]
 
    for json_path in json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)
            for shape in data['shapes']:
                name = shape['label']
                names.append(name)
 
    result = Counter(names)
    return result
 
 
def main(image_dir, json_dir, txt_dir, save_dir):
    # 创建文件夹
    mkdir(save_dir)
    images_dir = os.path.join(save_dir, 'images')
    labels_dir = os.path.join(save_dir, 'labels')
 
    img_train_path = os.path.join(images_dir, 'train')
    img_val_path = os.path.join(images_dir, 'val')
 
    label_train_path = os.path.join(labels_dir, 'train')
    label_val_path = os.path.join(labels_dir, 'val')
 
    mkdir(images_dir)
    mkdir(labels_dir)
    mkdir(img_train_path)
    mkdir(img_val_path)
    mkdir(label_train_path)
    mkdir(label_val_path)
 
    # 数据集划分比例，训练集75%，验证集15%，测试集15%，按需修改
    train_percent = 0.80
    val_percent = 0.20
 
    total_txt = os.listdir(txt_dir)
    num_txt = len(total_txt)
    list_all_txt = range(num_txt)  # 范围 range(0, num)
 
    num_train = int(num_txt * train_percent)
    num_val = int(num_txt * val_percent)
 
    train = random.sample(list_all_txt, num_train)
    # 在全部数据集中取出train
    val = [i for i in list_all_txt if not i in train]
    # 再从val_test取出num_val个元素，val_test剩下的元素就是test
    # val = random.sample(list_all_txt, num_val)
 
    print("训练集数目：{}, 验证集数目：{}".format(len(train), len(val)))
    for i in list_all_txt:
        name = total_txt[i][:-4]
 
        # srcImage = os.path.join(image_dir, name + '.jpg')
        srcImage = os.path.join(image_dir, name + '.png')
        srcLabel = os.path.join(txt_dir, name + '.txt')
 
        if i in train:
            # dst_train_Image = os.path.join(img_train_path, name + '.jpg')
            dst_train_Image = os.path.join(img_train_path, name + '.png')
            dst_train_Label = os.path.join(label_train_path, name + '.txt')
            shutil.copyfile(srcImage, dst_train_Image)
            shutil.copyfile(srcLabel, dst_train_Label)
        elif i in val:
            # dst_val_Image = os.path.join(img_val_path, name + '.jpg')
            dst_val_Image = os.path.join(img_val_path, name + '.png')
            dst_val_Label = os.path.join(label_val_path, name + '.txt')
            shutil.copyfile(srcImage, dst_val_Image)
            shutil.copyfile(srcLabel, dst_val_Label)
 
    obj_classes = get_classes(json_dir)
    classes = list(obj_classes.keys())
 
    # 编写yaml文件
    classes_txt = {i: classes[i] for i in range(len(classes))}  # 标签类别
    data = {
        'path': os.path.join(os.getcwd(), save_dir),
        'train': "images/train",
        'val': "images/val",
        'names': classes_txt,
        'nc': len(classes)
    }
    with open(save_dir + '/segment.yaml', 'w', encoding="utf-8") as file:
        yaml.dump(data, file, allow_unicode=True)
    print("标签：", dict(obj_classes))
 
 
if __name__ == "__main__":
    """
    python json2txt_nomalize.py --json-dir my_datasets/color_rings/jsons --save-dir my_datasets/color_rings/txts --classes "cat,dogs"
    """
    classes_list = 'arrow,capacitor,chip,ground,line,line_connector,motor,resistor,zener_diode,mov,fuse,inductor'  # 类名
 
    parser = argparse.ArgumentParser(description='json convert to txt params')
    parser.add_argument('--image-dir', type=str,default='imagessegment\\img', help='图片地址')
    parser.add_argument('--json-dir', type=str,default='imagessegment\\json', help='json地址')
    parser.add_argument('--txt-dir', type=str,default='imagessegment\\txt' , help='保存txt文件地址')
    parser.add_argument('--save-dir', default='imagessegment\\seg', type=str, help='保存最终分割好的数据集地址')
    parser.add_argument('--classes', type=str, default=classes_list, help='classes')
    args = parser.parse_args()
    json_dir = args.json_dir
    txt_dir = args.txt_dir
    image_dir = args.image_dir
    save_dir = args.save_dir
    classes = args.classes
    # json格式转txt格式
    convert_label_json(json_dir, txt_dir, classes)
    # 划分数据集，生成yaml训练文件
    main(image_dir, json_dir, txt_dir, save_dir)
