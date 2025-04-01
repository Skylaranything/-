import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def preprocess_image(image_path):
    # 1. 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read.")

    # 2. 去掉RGB像素值较低的蓝色噪声
    lower_blue_noise = np.array([0, 0, 100])  # 自定义蓝色噪声的下界
    upper_blue_noise = np.array([50, 50, 255])  # 自定义蓝色噪声的上界
    mask_noise = cv2.inRange(image, lower_blue_noise, upper_blue_noise)
    image[mask_noise > 0] = [0, 0, 0]

    # 3. 提取蓝色区域（标准蓝色范围：R=0, G=0, B=255）
    lower_blue = np.array([100, 0, 0])
    upper_blue = np.array([255, 100, 100])
    blue_mask = cv2.inRange(image, lower_blue, upper_blue)

    # 转换为灰度图
    blue_gray = cv2.bitwise_and(image, image, mask=blue_mask)
    gray_image = cv2.cvtColor(blue_gray, cv2.COLOR_BGR2GRAY)

    # 4. 归一化并二值化
    norm_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
    _, binary_image = cv2.threshold(norm_image, 50, 255, cv2.THRESH_BINARY)

    # 5. 连通区域标记
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # 按区域面积排序并筛选最大区域
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 排除背景区域
    largest_component = (labels == largest_label).astype(np.uint8) * 255

    # 6. 对最大区域进行闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed_image = cv2.morphologyEx(largest_component, cv2.MORPH_CLOSE, kernel)

    # 7. 再次进行连通区域标记，筛选最终最大区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_image, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    final_plate_region = (labels == largest_label).astype(np.uint8) * 255

    return final_plate_region

def extract_license_plate(image_path):
    preprocessed_image = preprocess_image(image_path)
    
    # 找到外轮廓
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 假设车牌是最大的矩形区域
    license_plate = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if 2 < aspect_ratio < 6:  # 车牌长宽比一般在这个范围内
            license_plate = (x, y, w, h)
            break

    if license_plate is not None:
        x, y, w, h = license_plate
        original_image = cv2.imread(image_path)
        plate_image = original_image[y:y+h, x:x+w]
        plate_image = cv2.resize(plate_image, (440, 140))
        return plate_image
    else:
        raise ValueError("License plate could not be detected.")

def save_image_with_matplotlib(image_array, output_path):
    plt.imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    print(f"Image saved successfully at {output_path}")

def process_images_in_folder(input_folder_path, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        
    image_count = 0
    for imgname in os.listdir(input_folder_path):
        if imgname.endswith('.jpg'):
            try:
                input_image_path = os.path.join(input_folder_path, imgname)
                plate_image = extract_license_plate(input_image_path)
                
                # 保存截取的车牌图像
                output_path = os.path.join(output_folder_path, f"extracted_{imgname}")
                save_image_with_matplotlib(plate_image, output_path)
                
                image_count += 1
                if image_count >= 10:
                    break
                
            except ValueError as e:
                print(f"Error processing {imgname}: {e}")
            
def get_image_folder_path():
    return "D:/code/Imgework/max_Imagework/1_class1_handle/test"  # 返回图像文件夹路径


def main():
    input_folder_path = "1_class1_handle/test"  # 输入文件夹路径
    output_folder_path = "1_cu_place/cu_result"  # 输出文件夹路径
    process_images_in_folder(input_folder_path, output_folder_path)

if __name__ == "__main__":
    main()