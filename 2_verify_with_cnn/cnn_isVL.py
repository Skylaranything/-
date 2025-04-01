import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

# 定义数据路径
good_folder = '2_verify_with_cnn/train/1'
bad_folder = '2_verify_with_cnn/train/0'

# 加载图像
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, (64, 64))
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)

# 加载正样本和负样本
good_images, good_labels = load_images_from_folder(good_folder, 1)
bad_images, bad_labels = load_images_from_folder(bad_folder, 0)

# 合并数据集
X = np.concatenate((good_images, bad_images), axis=0)
y = np.concatenate((good_labels, bad_labels), axis=0)

# 归一化图像数据
X = X.astype('float32') / 255.0

# 调整标签的形状
y = y.reshape(-1, 1)

# 使用 shuffle() 方法打乱数据集
X, y = shuffle(X, y, random_state=42)

# 显示加载的数据分布
print(f"数据加载完成：{len(good_images)} 个正样本，{len(bad_images)} 个负样本。")
print(f"打乱后的数据：{len(X)} 张图像，标签分布：{np.bincount(y.flatten())}")

# 创建模型
def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

# 定义输入形状并编译模型
input_shape = (64, 64, 3)
model = create_model(input_shape)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 数据增强
datagen = ImageDataGenerator(validation_split=0.2)

# 定义训练集和验证集生成器
train_gen = datagen.flow(X, y, batch_size=32, subset='training', shuffle=True)
val_gen = datagen.flow(X, y, batch_size=32, subset='validation', shuffle=True)

# 训练模型
history = model.fit(train_gen, validation_data=val_gen, epochs=20)

# 打印训练过程中的准确度
print("训练过程中的准确度：")
print(f"最终训练准确度: {history.history['accuracy'][-1]:.4f}")
print(f"最终验证准确度: {history.history['val_accuracy'][-1]:.4f}")

# 保存模型
model_save_path = "isVL_model.h5"  # 模型保存路径
model.save(model_save_path)
print(f"模型已保存至 {model_save_path}")

# 可以使用 model.evaluate() 评估测试集的准确度
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"测试集准确度: {test_accuracy:.4f}")
