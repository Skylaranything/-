import os
import numpy as np
import cv2 as cv
import tensorflow as tf
from keras import models, layers

# 数据目录路径
train_dir = "character_classification/chars"
test_dir = "character_classification/enu_test"
model_path = "character_classification/1230_en.h5"
image_width = 40
image_height = 40
classification_count = 34

# 标签字典
label_dict = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'J': 18, 'K': 19,
    'L': 20, 'M': 21, 'N': 22, 'P': 23, 'Q': 24, 'R': 25, 'S': 26, 'T': 27, 'U': 28, 'V': 29,
    'W': 30, 'X': 31, 'Y': 32, 'Z': 33
}

# 配置GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU 已启用！")
    except:
        print("GPU 配置失败！")
else:
    print("未检测到 GPU，将使用 CPU。")



class CnnModel:
    def __init__(self, image_height, image_width, classification_count):
        self.image_height = image_height
        self.image_width = image_width
        self.classification_count = classification_count
        self.model = None
       
        # 3个双卷积层块，引入了BatchNormalization层，双卷积层设计提供更好的特征学习，BatchNorm提供更稳定的训练
    def build_model(self):
        print('构建模型...')
        self.model = models.Sequential([
            # 第一个卷积块
            layers.Conv2D(32, (3, 3), padding='same', input_shape=(self.image_height, self.image_width, 1)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第二个卷积块
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第三个卷积块
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 全连接层
            layers.Flatten(),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.5),
            layers.Dense(self.classification_count, activation='softmax')
        ])
        
        # 设置优化器和学习率衰减，实现了学习率动态调整，更精细的优化器控制有助于模型收敛
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.summary()

    def load_data(self, dir_path):
        data = []
        labels = []
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    gray_image = cv.imread(subitem_path, cv.IMREAD_GRAYSCALE)
                    resized_image = cv.resize(gray_image, (self.image_width, self.image_height))
                    data.append(resized_image)
                    # 使用label_dict进行标签转换
                    labels.append(label_dict[item])
        return np.array(data), np.array(labels)

    def preprocess_data(self, train_dir, test_dir):
        print('加载数据...')
        train_data, train_labels = self.load_data(train_dir)
        test_data, test_labels = self.load_data(test_dir)
        
        # 数据归一化
        train_data = train_data / 255.0
        test_data = test_data / 255.0
        
        # One-hot编码
        train_labels = tf.keras.utils.to_categorical(train_labels, self.classification_count)
        test_labels = tf.keras.utils.to_categorical(test_labels, self.classification_count)
        
        # 调整维度
        train_data = train_data.reshape(-1, self.image_height, self.image_width, 1)
        test_data = test_data.reshape(-1, self.image_height, self.image_width, 1)
        
        return train_data, train_labels, test_data, test_labels

    def train(self, train_data, train_labels, test_data, test_labels, epochs=50, batch_size=32):
        print('开始训练...')
        
        # 数据增强
        data_augmentation = tf.keras.Sequential([
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1)
        ])
        
        # 早停策略
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # 模型检查点
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        
        history = self.model.fit(
            train_data, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(test_data, test_labels),
            callbacks=[early_stopping, checkpoint],
            shuffle=True
        )
        
        return history

    def evaluate(self, test_data, test_labels):
        print('评估模型...')
        loss, accuracy = self.model.evaluate(test_data, test_labels)
        print(f'测试准确率：{accuracy * 100:.2f}%')
        return accuracy
    def save_model(self, path):
        print('保存模型...')
        self.model.save(path)


if __name__ == '__main__':
    cnn = CnnModel(image_height, image_width, classification_count)
    train_data, train_labels, test_data, test_labels = cnn.preprocess_data(train_dir, test_dir)
    train_data, train_labels, test_data, test_labels = cnn.preprocess_data(train_dir, test_dir)
    # 构建模型
    cnn.build_model()
    
    # 训练模型
    history = cnn.train(train_data, train_labels, test_data, test_labels, epochs=50, batch_size=32)
    
    # 评估模型
    accuracy = cnn.evaluate(test_data, test_labels)
    
    # 保存模型
    cnn.save_model(model_path)
    print(f'模型测试准确率：{accuracy * 100:.2f}%')
