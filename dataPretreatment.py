# 生成训练、验证、测试集
import os
from PIL import Image
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from paddle.vision.transforms import Compose, Grayscale, Transpose, BrightnessTransform, Resize, Normalize, \
    RandomHorizontalFlip, RandomRotation, ContrastTransform, RandomCrop
from paddle.io import DataLoader, Dataset


# 从mask找出裁剪的四角
def find_anchor(maskimg):
    maskimg = maskimg.tolist()
    # y,x,w,h
    h = 0
    w = 0
    y = 0
    x = 0
    for i in range(len(maskimg)):
        for j in range(len(maskimg[0])):
            if maskimg[i][j] == 255:
                w += 1
                if y == 0:
                    y = i
                    x = j
        if not w == 0:
            break
    for i in range(y, len(maskimg)):
        if maskimg[i][x] == 255:
            h += 1
    return y, x, w, h


def generate_dataset(dataset_path, mode="train"):
    # dataset_path = "/home/aistudio/work/original/"
    data_path = os.path.join(dataset_path, "data/")
    mask_path = os.path.join(dataset_path, "mask/")
    if mode == "train":
        masked_path = "work/masked_train/"
    else:
        masked_path = "work/masked_test/"
    csv_path = os.path.join(dataset_path, "label.csv")

    trainf = open(os.path.join("work/", 'train_list.txt'), 'a')
    valf = open(os.path.join("work/", 'val_list.txt'), 'a')
    testf = open(os.path.join("work/", 'test_list.txt'), 'a')

    # 蒙皮词典
    mask_dict = {}
    with open(csv_path) as f:
        csv_file = csv.reader(f)
        lines = list(csv_file)
        lines = lines[1:]
    random.shuffle(lines)
    for idx, line in enumerate(lines):
        # lineList = line[0:-1].split('\t',1)
        imgname = line[0]
        imgname = imgname[0:-3] + "jpg"
        # print(imgname)
        img_dir = os.path.join(data_path, imgname)
        # 10_0.jpg->10_0_mask.jpg
        mask_dir = os.path.join(mask_path, imgname[:-4] + "_mask.jpg")
        # print(mask_dir)
        masked_dir = os.path.join(masked_path, imgname)
        value = line[1]
        mask_dict[img_dir] = mask_dir

        # 只有没有的时候生成
        if not os.path.exists(masked_dir):
            # 利用mask，生成蒙皮图象
            image = cv2.imread(img_dir)
            if image is None:
                print(img_dir)
                # return
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            maskimg = cv2.imread(mask_dir)
            maskimg = cv2.cvtColor(maskimg, cv2.COLOR_BGR2GRAY)
            # y,x,w,h
            y, x, w, h = find_anchor(maskimg)
            # print(image.shape)
            # print(y,x,w,h)
            image2 = image[y:y + h, x:x + w].copy()
            # maskimg.reshape(image.shape)
            # image2 = cv2.add(image, np.zeros(np.shape(image), dtype=np.uint8), mask=maskimg)
            # image2 = cv2.bitwise_and(image, maskimg)
            cv2.imwrite(masked_dir, image2)

        if mode == "test":
            testf.write((masked_dir + ' ' + str(value) + '\n'))

        else:
            if idx % 10 == 0:
                valf.write((masked_dir + ' ' + str(value) + '\n'))
            # elif idx % 9 == 0:
            #     testf.write((masked_dir + ' ' + str(value) + '\n'))
            else:
                # 重采样
                if value == 1:
                    trainf.write((masked_dir + ' ' + str(value) + '\n'))
                    trainf.write((masked_dir + ' ' + str(value) + '\n'))
                trainf.write((masked_dir + ' ' + str(value) + '\n'))

    trainf.close()
    # maskedtrainf.close()
    valf.close()
    testf.close()
    print('finished!')


def generate_data_files(train_path_list, test_path_list, save_dir="work/"):
    for i in train_path_list:
        generate_dataset(i, "train")
    for i in test_path_list:
        generate_dataset(i, "test")

    # 对三个文件做打乱
    with open(os.path.join(save_dir, 'train_list.txt'), 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open(os.path.join(save_dir, 'train_list.txt'), 'w') as f:
        for line in lines:
            f.write(line)

    with open(os.path.join(save_dir, 'val_list.txt'), 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open(os.path.join(save_dir, 'val_list.txt'), 'w') as f:
        for line in lines:
            f.write(line)

    with open(os.path.join(save_dir, 'test_list.txt'), 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open(os.path.join(save_dir, 'test_list.txt'), 'w') as f:
        for line in lines:
            f.write(line)


# 定义DataSet
class XChestDateset(Dataset):
    def __init__(self, txt_path, transform=None, mode='train'):
        super(XChestDateset, self).__init__()
        self.mode = mode
        self.data_list = []
        self.transform = transform

        if mode == 'train':
            self.data_list = np.loadtxt(txt_path, dtype='str')
        elif mode == 'valid':
            self.data_list = np.loadtxt(txt_path, dtype='str')
        elif mode == 'test':
            self.data_list = np.loadtxt(txt_path, dtype='str')

    def __getitem__(self, idx):
        img_path = self.data_list[idx][0]
        img = cv2.imread(img_path)
        # 处理错误数据
        if img is None:
            print(img_path)
            with open("wrong_data.txt", "a") as f:
                f.write(img_path + "/n")
            return self.__getitem__(idx - 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, np.array(self.data_list[idx][1]).astype('int64')

    def __len__(self):
        return self.data_list.shape[0]


# 预处理，增广，生成数据集
def generate_dataloader(train_txt="work/train_list.txt", test_txt="work/test_list.txt", val_txt="work/val_list.txt",
                        BATCH_SIZE=64):
    train_transform = Compose([RandomRotation(degrees=180),  # 随机旋转0到10度
                               RandomHorizontalFlip(),  # 随机翻转
                               ContrastTransform(0.1),  # 随机调整图片的对比度
                               BrightnessTransform(0.1),  # 随机调整图片的亮度
                               Grayscale(),  # 灰度化，因为超声图像颜色其实没意义
                               # 换成图象处理的时候直接转灰度
                               # Resize(size=(240,240)),#调整图片大小为240,240
                               # RandomCrop(size=(224,224)),#从240大小中随机裁剪出224
                               Resize(size=(224, 224)),
                               Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC'),
                               # 归一化
                               Transpose()])  # 对‘HWC’转换成‘CHW’

    val_transform = Compose([Grayscale(),
                             Resize(size=(224, 224)),
                             Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC'),
                             Transpose()])
    trn_dateset = XChestDateset(train_txt, train_transform, 'train')
    train_loader = DataLoader(trn_dateset, shuffle=True, batch_size=BATCH_SIZE)
    val_dateset = XChestDateset(val_txt, val_transform, 'valid')
    valid_loader = DataLoader(val_dateset, shuffle=False, batch_size=BATCH_SIZE)
    test_dateset = XChestDateset(test_txt, val_transform, 'valid')
    test_loader = DataLoader(test_dateset, shuffle=False, batch_size=BATCH_SIZE)
    print(len(trn_dateset))
    print(len(val_dateset))
    return train_loader, valid_loader, test_loader


# 可视化观察
def imshow(img):
    img = np.transpose(img, (1, 2, 0))
    img = img * 127.5 + 127.5  # 反归一化，还原图片
    img = img.astype(np.int32)
    plt.imshow(img)


def preview(train_loader):
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    num = images.shape[0]
    print(num)
    row = 4
    fig = plt.figure(figsize=(14, 14))

    for idx in range(num):
        ax = fig.add_subplot(row, int(num / row), idx + 1, xticks=[], yticks=[])
        imshow(images[idx])
        if labels[idx]:
            ax.set_title('BA')
        else:
            ax.set_title('non-BA')


def main():
    # 所有文件路径,看情况修改
    train_path_list = ["work/Original_train_dataset", "work/MobilePhone_train_dataset/doctor_A",
                       "work/MobilePhone_train_dataset/doctor_C", "work/MobilePhone_train_dataset/doctor_D",
                       "work/MobilePhone_train_dataset/doctor_E", "work/MobilePhone_train_dataset/doctor_F"]
    test_path_list = ["work/Original_test_dataset", "work/MobilePhone_test_dataset/doctorA",
                      "work/MobilePhone_test_dataset/doctorB", "work/MobilePhone_test_dataset/doctorC",
                      "work/MobilePhone_test_dataset/doctorD", "work/MobilePhone_test_dataset/doctorE",
                      "work/MobilePhone_test_dataset/doctorF", "work/MobilePhone_test_dataset/doctorG"]

    generate_data_files(train_path_list, test_path_list)

    # 文件地址
    train_txt = "work/train_list.txt"
    test_txt = "work/test_list.txt"
    val_txt = "work/val_list.txt"

    train_loader, valid_loader, test_loader = generate_dataloader(BATCH_SIZE=64)

    preview(train_loader)


if __name__ == "__main__":
    main()
