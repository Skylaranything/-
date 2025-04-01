from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot

provincelist = [
    "皖", "沪", "津", "渝", "冀",
    "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云",
    "西", "陕", "甘", "青", "宁",
    "新"]

wordlist = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9"]

# --- 绘制边界框
def DrawBox(im, box):
    draw = ImageDraw.Draw(im)
    draw.rectangle([tuple(box[0]), tuple(box[1])], outline="#FFFFFF", width=3)

# --- 绘制四个关键点
def DrawPoint(im, points):
    draw = ImageDraw.Draw(im)
    for p in points:
        center = (p[0], p[1])
        radius = 5
        right = (center[0] + radius, center[1] + radius)
        left = (center[0] - radius, center[1] - radius)
        draw.ellipse((left, right), fill="#FF0000")

# --- 绘制车牌
def DrawLabel(im, label):
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype('simsun.ttc', 64)
    draw.text((30, 30), label, font=font, fill="#FF0000")  # 使用红色字体

# --- 图片可视化
def ImgShow(imgpath, box, points, label, output_path):
    # 打开图片
    im = Image.open(imgpath)
    DrawBox(im, box)
    DrawPoint(im, points)
    DrawLabel(im, label)
    # 保存并显示图片
    im.save(output_path)
    im.show()

def extract_license_plate(imgpath, points, output_path):
    img = cv2.imread(imgpath)
    
    # 检查图像是否正确读取
    if img is None:
        raise ValueError(f"Failed to load image at path: {imgpath}")

    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [440, 0], [440, 140], [0, 140]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (440, 140))
    cv2.imwrite(output_path, dst)
    return dst

def main():
    # 文件夹路径
    folder_path = '1_class1_handle/CCPD(1650)'
    image_count = 0  # 初始化计数器

    # 遍历文件夹中的所有图像文件
    for imgname in os.listdir(folder_path):
        if imgname.endswith('.jpg'):
            imgpath = os.path.join(folder_path, imgname)

            # 根据图像名分割标注
            _, _, box, points, label, brightness, blurriness = imgname.split('-')

            # --- 边界框信息
            box = box.split('_')
            box = [list(map(int, i.split('&'))) for i in box]

            # --- 关键点信息
            points = points.split('_')
            points = [list(map(int, i.split('&'))) for i in points]
            # 将关键点的顺序变为从左上顺时针开始
            points = points[-2:] + points[:2]

            # --- 读取车牌号
            label = label.split('_')
            # 省份缩写
            province = provincelist[int(label[0])]
            # 车牌信息
            words = [wordlist[int(i)] for i in label[1:]]
            # 车牌号
            label = province + ''.join(words)

            # --- 图片可视化
            output_path = os.path.join(folder_path, 'result_' + imgname)
            ImgShow(imgpath, box, points, label, output_path)

            # --- 提取车牌区域
            plate_output_path = os.path.join(folder_path, 'extracted_' + imgname)
            extracted_plate = extract_license_plate(imgpath, points, plate_output_path)
            plt.imshow(cv2.cvtColor(extracted_plate, cv2.COLOR_BGR2RGB))
            plt.title('Extracted License Plate')
            plt.axis('off')
            plt.show()
            
            image_count += 1  # 增加计数器
            if image_count >= 2:  # 达到10张图像后停止
                break

main()