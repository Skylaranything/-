import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from keras import models, layers
from sklearn.model_selection import train_test_split

# 数据目录路径
train_dir = "character_classification/charsChinese"
model_path = "character_classification/12300/1230_ch.h5"
image_width = 40
image_height = 40
class_count = 31
label_map = {
    'chuan': 0, 'e': 1, 'gan': 2, 'gan1': 3, 'gui': 4, 'gui1': 5, 'hei': 6, 'hu': 7, 'ji': 8, 'jin': 9,
    'jing': 10, 'jl': 11, 'liao': 12, 'lu': 13, 'meng': 14, 'min': 15, 'ning': 16, 'qing': 17, 'qiong': 18, 'shan': 19,
    'su': 20, 'sx': 21, 'wan': 22, 'xiang': 23, 'xin': 24, 'yu': 25, 'yu1': 26, 'yue': 27, 'yun': 28, 'zang': 29,
    'zhe': 30
}

# 配置GPU
physical_devices = tf.config.list_physical_devices('gpu')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

class CnnModel:
    def __init__(self):
        self.model = None
    # 模型B的特征图数量是模型A的2倍，模型B的参数量更大，特征提取能力更强，模型B采用了更全面的正则化策略，逐层增加Dropout率
    def build(self):
        self.model = models.Sequential([
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=(image_height, image_width, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(class_count, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def load_data(self, directory):
        images, labels = [], []
        for folder in os.listdir(directory):
            folder_path = os.path.join(directory, folder)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                    img_resized = cv.resize(img, (image_width, image_height))
                    images.append(img_resized.ravel())
                    labels.append(label_map[folder])
        return np.array(images), np.array(labels)

    def one_hot_encode(self, labels):
        one_hot = np.zeros((len(labels), class_count))
        one_hot[np.arange(len(labels)), labels] = 1
        return one_hot

    def preprocess_data(self):
        # 加载数据
        data, labels = self.load_data(train_dir)
        data = (data - data.mean()) / data.max()  # 数据归一化

        # 划分训练集和验证集（80%训练，20%验证）
        train_data, val_data, train_labels, val_labels = train_test_split(
            data, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # 独热编码标签
        train_labels = self.one_hot_encode(train_labels)
        val_labels = self.one_hot_encode(val_labels)

        # 调整数据维度以适配 CNN 输入
        self.train_data = tf.reshape(train_data, [-1, image_height, image_width, 1])
        self.val_data = tf.reshape(val_data, [-1, image_height, image_width, 1])
        self.train_labels = train_labels
        self.val_labels = val_labels

        print('数据加载与划分完成')

    def train(self, epochs=50):
        self.model.fit(self.train_data, self.train_labels, epochs=epochs, shuffle=True, validation_data=(self.val_data, self.val_labels))

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.val_data, self.val_labels)
        print(f'验证集准确率：{accuracy * 100:.2f}%')

    def save(self, path=model_path):
        self.model.save(path)
        print(f'模型保存到：{path}')

    def load(self, path=model_path):
        self.model = models.load_model(path)
        print(f'模型加载自：{path}')

if __name__ == '__main__':
    cnn = CnnModel()
    cnn.preprocess_data()
    cnn.build()
    cnn.train()
    cnn.evaluate()
    cnn.save()