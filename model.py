import tensorflow as tf
import numpy as np
import os
from PIL import Image


# 数据加载和预处理
class LicensePlateDataset:
    def __init__(self, chars_path, chars_chinese_path):
        self.image_paths = []
        self.labels = []
        self.label_map = {}
        current_label = 0

        # 处理数字和字母
        for char_folder in sorted(os.listdir(chars_path)):
            self.label_map[char_folder] = current_label
            folder_path = os.path.join(chars_path, char_folder)
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(current_label)
            current_label += 1

        # 处理中文字符
        for char_folder in sorted(os.listdir(chars_chinese_path)):
            self.label_map[char_folder] = current_label
            folder_path = os.path.join(chars_chinese_path, char_folder)
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(current_label)
            current_label += 1

        self.num_classes = len(self.label_map)

    def load_and_preprocess_image(self, image_path):
        # 读取图片并转换为灰度图
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [40, 40])
        img = tf.cast(img, tf.float32) / 255.0
        img = (img - 0.5) / 0.5  # 归一化到[-1, 1]
        return img

    def create_dataset(self, batch_size=32):
        # 创建tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.image_paths, self.labels)
        )
        dataset = dataset.map(
            lambda x, y: (self.load_and_preprocess_image(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


# 创建CNN模型
def create_model(num_classes):
    model = tf.keras.Sequential([
        # 第一个卷积块
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(40, 40, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # 第二个卷积块
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # 第三个卷积块
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        # 第四个卷积块
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.3),

        # 全连接层
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


# 训练模型
def train_model():
    # 创建数据集
    dataset = LicensePlateDataset(
        chars_path='train1225/chars',
        chars_chinese_path='train1225/charsChinese'
    )
    train_ds = dataset.create_dataset(batch_size=32)

    # 创建模型
    model = create_model(dataset.num_classes)

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 学习率调度器
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        verbose=1
    )

    # 训练模型
    model.fit(
        train_ds,
        epochs=80,
        callbacks=[reduce_lr]
    )

    # 保存模型和标签映射
    model.save('license_plate_model')
    np.save('label_map.npy', dataset.label_map)

    return model, dataset.label_map


# 预测函数
def predict_images(image_folder, model, label_map):
    reverse_label_map = {v: k for k, v in label_map.items()}
    results = []

    # 读取并预处理图片
    for img_name in sorted(os.listdir(image_folder))[:7]:
        img_path = os.path.join(image_folder, img_name)
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [40, 40])
        img = tf.cast(img, tf.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = tf.expand_dims(img, 0)  # 添加batch维度

        # 预测
        predictions = model.predict(img)
        predicted_label = np.argmax(predictions[0])
        predicted_char = reverse_label_map[predicted_label]
        results.append(predicted_char)

    return results


# 使用示例
def main():
    # 训练模型
    model, label_map = train_model()

    # 预测新图片
    test_folder = "plate/1"  # 替换为测试图片文件夹路径
    results = predict_images(test_folder, model, label_map)
    print("预测结果:", results)


if __name__ == "__main__":
    main()
