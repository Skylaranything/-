import os
import cv2
import numpy as np
from keras.models import load_model

# 定义文件夹路径
test_folder = '1_cu_place/cu_result'  # 测试图像文件夹路径
yes_folder = '2_verify_with_cnn/result/yes'
no_folder = '2_verify_with_cnn/result/no'

# 加载保存的模型
model = load_model('2_verify_with_cnn/isVL_model.h5')

# 定义图像分类和保存的函数
def classify_and_save_images(model, folder, yes_folder, no_folder):
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            image = cv2.imread(img_path)
            if image is not None:
                image_resized = cv2.resize(image, (64, 64))
                image_resized = image_resized.astype('float32') / 255.0
                image_resized = np.expand_dims(image_resized, axis=0)
                
                # 使用模型进行预测
                prediction = model.predict(image_resized)
                
                # 根据预测结果保存图像
                if prediction > 0.5:
                    save_path = os.path.join(yes_folder, filename)
                else:
                    save_path = os.path.join(no_folder, filename)
                
                cv2.imwrite(save_path, image)
                print(f"图像 {filename} 被分类为 {'车牌' if prediction > 0.5 else '非车牌'} 并保存到 {save_path}")

# 封装代码到 main 函数
def main():
    # 如果目标文件夹不存在，则创建
    if not os.path.exists(yes_folder):
        os.makedirs(yes_folder)
    if not os.path.exists(no_folder):
        os.makedirs(no_folder)

    # 对测试文件夹中的图像进行分类并保存
    classify_and_save_images(model, test_folder, yes_folder, no_folder)

# 确保主函数只在脚本直接运行时调用
if __name__ == "__main__":
    main()