import tensorflow as tf
import numpy as np
import os
from PIL import Image


def load_and_predict_detailed(test_folder_path, confidence_threshold=0.5):
    # 加载模型和标签映射
    print("正在加载模型和标签映射...")
    model = tf.keras.models.load_model('license_plate_model')
    label_map = np.load('label_map.npy', allow_pickle=True).item()
    reverse_label_map = {v: k for k, v in label_map.items()}

    print(f"\n找到 {len(os.listdir(test_folder_path))} 张图片待识别")
    results = []

    for img_name in sorted(os.listdir(test_folder_path)):
        print(f"\n处理图片: {img_name}")
        img_path = os.path.join(test_folder_path, img_name)

        # 读取和预处理图片
        try:
            img = tf.io.read_file(img_path)
            img = tf.image.decode_png(img, channels=1)
            img = tf.image.resize(img, [40, 40])
            img = tf.cast(img, tf.float32) / 255.0
            img = (img - 0.5) / 0.5
            img = tf.expand_dims(img, 0)

            # 预测
            predictions = model.predict(img, verbose=0)
            predicted_label = np.argmax(predictions[0])
            confidence = predictions[0][predicted_label]
            predicted_char = reverse_label_map[predicted_label]

            # 输出预测结果和置信度
            print(f"预测字符: {predicted_char}")
            print(f"置信度: {confidence:.2%}")

            if confidence < confidence_threshold:
                print(f"警告: 置信度低于阈值 ({confidence_threshold:.2%})")

            results.append({
                'image': img_name,
                'predicted': predicted_char,
                'confidence': confidence
            })

        except Exception as e:
            print(f"处理图片 {img_name} 时出错: {str(e)}")
            continue

    return results

def get_province_map():
    """返回省份简称到全称的映射字典"""
    return {
        'wan': '皖',
        'chuan': '川',
        'sx': '晋',
        'gan': '赣',
        'lu': '鲁',
        'yue': '粤',
        'yu1': '渝',
        'min': '闽',
        'zhe': '浙',
        'su': '苏',
        'yu': '豫',
        'jl': '吉',
        'yun': '云',
        'xiang': '湘',
        'qing': '青',
        'gui': '贵',
        'qiong': '琼',
        'zang': '藏',
        'e': '鄂',
        'ning': '宁',
        'jin': '津',
        'jing': '京',
        'liao': '辽',
        'hu': '沪',
        'xin': '新',
        'gan1': '甘',
        'shan': '陕',
        'meng': '蒙',
        'gui1': '桂',
        'hei': '黑',
        'ji': '冀'
    }

def print_results_summary(results):
    print("\n=== 识别结果汇总 ===")
    print(f"总计处理图片: {len(results)} 张")

    # 提取字符序列
    chars = [r['predicted'] for r in results]

    # 转换第一个字符（如果存在）
    if chars:
        province_map = get_province_map()
        first_char = chars[0].lower()
        if first_char in province_map:
            chars[0] = province_map[first_char]

    print("\n完整字符序列:", chars)

    # 计算平均置信度
    avg_confidence = np.mean([r['confidence'] for r in results])
    print(f"\n平均置信度: {avg_confidence:.2%}")

    # 输出详细结果
    print("\n详细结果:")
    for r in results:
        print(f"图片: {r['image']:<20} 预测: {r['predicted']:<5} 置信度: {r['confidence']:.2%}")


def process_all_subfolders(base_folder):
    """处理基础文件夹下的所有子文件夹"""
    all_results = {}

    # 获取所有子文件夹
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    print(f"找到 {len(subfolders)} 个子文件夹待处理")

    # 处理每个子文件夹
    for folder in sorted(subfolders):
        folder_name = os.path.basename(folder)
        print(f"\n开始处理子文件夹: {folder_name}")
        try:
            results = load_and_predict_detailed(folder)
            all_results[folder_name] = results
            print_results_summary(results)
        except Exception as e:
            print(f"处理文件夹 {folder_name} 时出错: {str(e)}")
            continue

    return all_results


def main():
    # 指定基础文件夹路径
    base_folder = "plate"  # 替换为您的基础文件夹路径

    try:
        print(f"开始处理基础文件夹: {base_folder}")
        all_results = process_all_subfolders(base_folder)

        # 打印所有文件夹的汇总结果
        print("\n\n=== 所有文件夹处理完成 ===")
        print(f"总计处理文件夹数: {len(all_results)}")

        # 可以在这里添加更多的汇总统计信息
        for folder_name, results in all_results.items():
            chars = [r['predicted'] for r in results]
            if chars:
                province_map = get_province_map()
                first_char = chars[0].lower()
                if first_char in province_map:
                    chars[0] = province_map[first_char]
            print(f"\n文件夹 {folder_name} 识别结果:", chars)

    except Exception as e:
        print(f"程序执行出错: {str(e)}")


if __name__ == "__main__":
    main()
