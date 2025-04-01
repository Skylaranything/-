import os
import numpy as np
import cv2 as cv
import tensorflow as tf
from keras import models

# 设定图像尺寸和标签
ch_image_width = 40
ch_image_height = 40
en_image_width = 40
en_image_height = 40

ch_label = [
    "川", "鄂", "赣", "甘", "贵", "桂", "黑", "沪", "冀", "津",
    "京", "吉", "辽", "鲁", "蒙", "闽", "宁", "青", "琼", "陕",
    "苏", "晋", "皖", "湘", "新", "豫", "渝", "粤", "云", "藏", "浙",
]
en_label = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z',
]

# 加载模型
model_ch = models.load_model("4_character_classification/model/Net1/1230_cn.h5")
model_en = models.load_model("4_character_classification/model/Net1/1230_en.h5")

def preprocess_image(image, target_width, target_height):
    resized_image = cv.resize(image, (target_width, target_height))
    normalized_image = (resized_image - resized_image.mean()) / resized_image.max()
    return tf.reshape(normalized_image, (-1, target_height, target_width, 1))

def predict_ch(gray_image):
    image = preprocess_image(gray_image, ch_image_width, ch_image_height)
    prediction = model_ch.predict(image)
    prediction_index = np.argmax(prediction)
    prediction_results = ch_label[prediction_index]
    return prediction_results, prediction[0][prediction_index]

def predict_en(gray_image):
    image = preprocess_image(gray_image, en_image_width, en_image_height)
    prediction = model_en.predict(image)
    prediction_index = np.argmax(prediction)
    prediction_results = en_label[prediction_index]
    return prediction_results, prediction[0][prediction_index]

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(image)
    return images

def recognize_characters_in_subfolder(subfolder):
    chars_image = load_images_from_folder(subfolder)
    char_result = []
    confidence_result = []

    if chars_image:
        char, confidence = predict_ch(chars_image[0])
        char_result.append(char)
        confidence_result.append(confidence)
        
        for i in range(1, len(chars_image)):
            char, confidence = predict_en(chars_image[i])
            char_result.append(char)
            confidence_result.append(confidence)
    
    return ''.join(char_result), confidence_result

def recognize_characters_from_folder(folder_path):
    results = {}
    for subfolder in sorted(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            result, confidence = recognize_characters_in_subfolder(subfolder_path)
            results[subfolder] = {'result': result, 'confidence': confidence}
    return results
    
def main():
    # 设置文件夹路径
    parent_folder_path = '3_split/outcut'

    # 识别字符
    all_results = recognize_characters_from_folder(parent_folder_path)

    for subfolder, res in all_results.items():
        print(f"子文件夹: {subfolder}")
        print("识别结果：", res['result'])
        print("置信度：", res['confidence'])

# 确保主函数只在脚本直接运行时调用
if __name__ == "__main__":
    main()