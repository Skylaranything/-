import threading
from queue import Queue
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import datetime
import cv2
import numpy as np
import class1_rcg
import Coarse_positioning
import recognize_l
import recognize_VL
import split_single
import straight_model1

class LicensePlateRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("车牌图像识别系统")

        # 添加处理队列
        self.process_queue = Queue()
        self.result_queue = Queue()
    
        # 启动处理线程
        self.processing_thread = None

        # 初始化属性
        self.current_image_index = 0
        self.image_files = []
        self.folder_path = ""
        # 添加缓存字典
        self.cache = {
            'processed_images': {},  # 存储处理后的原始图像
            'plate_images': {},      # 存储车牌区域图像
            'combined_images': {},   # 存储字符分割图像
            'plate_numbers': {},     # 存储车牌号
        }
        self.current_process_type = None  # 记录当前处理类型

        self.correct_count = 0  # 新增：正确率统计
        self.total_count = 0
        self.selected_model = None  # 新增：模型选择
        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        # 主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # 设置窗口大小
        self.root.geometry("1000x800")

        # 左侧框架
        left_frame = tk.Frame(main_frame, width=500)
        left_frame.pack(side="left", padx=20, fill="both", expand=True)

        # 原始图片显示区域
        self.image_label = tk.Label(left_frame, text="原始图片", relief="solid", bg="#d9d9d9", height=20)
        self.image_label.pack(pady=10, fill="both", expand=True)

        # 信息显示标签
        self.info_label = tk.Label(left_frame, text="请选择图片文件夹开始识别", font=("Arial", 12))
        self.info_label.pack(pady=10)

        # 导航按钮框架
        nav_frame = tk.Frame(left_frame)
        nav_frame.pack(pady=10, fill="x")

        self.prev_button = tk.Button(nav_frame, text="上一张", command=self.show_previous, state="disabled")
        self.prev_button.pack(side="left", expand=True, padx=5)

        self.next_button = tk.Button(nav_frame, text="下一张", command=self.show_next, state="disabled")
        self.next_button.pack(side="right", expand=True, padx=5)

        # 控制面板框架
        control_frame = tk.Frame(left_frame)
        control_frame.pack(pady=10, fill="x")

        # 选择测试类型
        self.option_var = tk.StringVar()
        self.option_menu = ttk.Combobox(control_frame, textvariable=self.option_var)
        self.option_menu['values'] = ("I类车牌集中测试", "II类车牌集中测试")
        self.option_menu.set("选择测试类型")
        self.option_menu.pack(pady=5, fill="x")


        # 添加模型选择（在测试类型选择之后）
        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(control_frame, textvariable=self.model_var)
        self.model_combobox['values'] = ("L1", "L2")
        self.model_combobox.set("选择识别模型")
        self.model_combobox.pack(pady=5, fill="x")
        self.model_combobox.bind("<<ComboboxSelected>>", self.on_model_selected)

        # 按钮区域
        button_frame = tk.Frame(control_frame)
        button_frame.pack(pady=5, fill="x")

        self.load_button = tk.Button(button_frame, text="加载文件夹", command=self.load_folder)
        self.load_button.pack(side="left", expand=True, padx=5)

        self.start_button = tk.Button(button_frame, text="开始识别", command=self.start_recognition)
        self.start_button.pack(side="right", expand=True, padx=5)

        # 右侧框架
        right_frame = tk.Frame(main_frame, width=500)
        right_frame.pack(side="right", padx=20, fill="both", expand=True)


        # 为下拉框绑定事件
        #self.path_combobox.bind("<<ComboboxSelected>>", self.update_path)

        # 车牌区域显示
        self.plate_image_label = tk.Label(right_frame, text="车牌区域", relief="solid", bg="#d9d9d9", height=10)
        self.plate_image_label.pack(pady=10, fill="both", expand=True)

        # 在车牌区域和识别结果之间新增一个区域
        self.combined_label = tk.Label(right_frame, text="拼接图像区域", relief="solid", bg="#d9d9d9", height=5)
        self.combined_label.pack(pady=10, fill="both", expand=True)

        # 识别结果显示
        result_frame = tk.Frame(right_frame)
        result_frame.pack(pady=10, fill="both", expand=True)

        self.result_label = tk.Label(result_frame, text="识别结果", font=("Arial", 12), justify="left")
        self.result_label.pack(pady=5)

        self.plate_text_label = tk.Label(result_frame, text="车牌号码：", font=("Arial", 14, "bold"))
        self.plate_text_label.pack(pady=5)
        # 在result_frame中添加字符对比和正确率显示
        self.comparison_label = tk.Label(result_frame, text="字符对比：", font=("Arial", 12))
        self.comparison_label.pack(pady=5)

        self.accuracy_label = tk.Label(result_frame, text="正确率：0%", font=("Arial", 12))
        self.accuracy_label.pack(pady=5)
    
        
    def on_model_selected(self, event=None):
        """处理模型选择事件"""
        self.selected_model = self.model_var.get()
        self.info_label.config(text=f"已选择模型: {self.selected_model}")


    def load_folder(self):
        """加载图片文件夹"""
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.folder_path = folder_path
            self.image_files = [f for f in os.listdir(folder_path)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            if self.image_files:
                self.current_image_index = 0
                self.show_current_image()
                self.update_navigation_buttons()
                self.info_label.config(text="文件夹加载成功")
            else:
                self.info_label.config(text="所选文件夹中没有图片文件")
    
    def show_current_image(self):
        """显示当前图片"""
        if self.image_files:
            image_path = os.path.join(self.folder_path, self.image_files[self.current_image_index])
            try:
                # 仅在图像内容变化时更新界面
                if self.image_label.image is None or self.image_label.image != image_path:
                    image = Image.open(image_path)
                    image.thumbnail((450, 450))
                    photo = ImageTk.PhotoImage(image)
                    self.image_label.config(image=photo, text="")
                    self.image_label.image = photo
                # 更新图片信息
                self.info_label.config(text=f"当前图片: {self.image_files[self.current_image_index]}")
            except Exception as e:
                self.info_label.config(text=f"无法加载图片: {e}")

    def show_previous(self):
        """显示上一张图片"""
        if self.current_image_index >0:
            self.current_image_index -= 1
            current_image = self.image_files[self.current_image_index]
            # 检查缓存中是否有结果
            if current_image in self.cache['processed_images']:
                self.show_current_image()
                self.display_cached_results(current_image)
            else:
                self.show_current_image()
                if self.current_process_type == "I类车牌集中测试":
                    self.process_class1_image()
                elif self.current_process_type == "II类车牌集中测试":
                    self.process_class2_image()
        
            self.update_navigation_buttons()

    def process_current_image(self):
        """处理当前图片"""
        selected_option = self.option_var.get()
        if selected_option == "I类车牌集中测试":
            return self.process_class1_image()
        elif selected_option == "II类车牌集中测试":
            return self.process_class2_image()
        return False

    def show_next(self):
        """显示下一张图片"""
        if self.current_image_index < len(self.image_files) - 1:
            # 如果有处理线程在运行，等待其完成
            if self.processing_thread and self.processing_thread.is_alive():
                self.info_label.config(text="请等待当前处理完成...")
                return

            self.current_image_index += 1
            current_image = self.image_files[self.current_image_index]
            # 检查缓存中是否有结果
            if current_image in self.cache['processed_images']:
                self.show_current_image()
                self.display_cached_results(current_image)
            else:
                self.show_current_image()
                if self.current_process_type == "I类车牌集中测试":
                    self.process_class1_image()
                elif self.current_process_type == "II类车牌集中测试":
                    self.process_class2_image()
                
            self.update_navigation_buttons()

    def update_navigation_buttons(self):
        """更新导航按钮状态"""
        self.prev_button.config(state="normal" if self.current_image_index > 0 else "disabled")
        self.next_button.config(state="normal"
                              if self.current_image_index < len(self.image_files) - 1
                              else "disabled")

    def process_class1_image(self):
        """处理I类车牌图像"""
        try:
            current_image = self.image_files[self.current_image_index]
            imgpath = os.path.join(self.folder_path, current_image)

            # 更新处理状态
            self.info_label.config(text="正在处理图像...")
            self.root.update()

            # 解析图像文件名获取信息
            try:
                _, _, box_info, points_info, label_info, brightness, blurriness = current_image.split('-')

                # 解析边界框信息
                box = box_info.split('_')
                box = [list(map(int, i.split('&'))) for i in box]

                # 解析关键点信息
                points = points_info.split('_')
                points = [list(map(int, i.split('&'))) for i in points]
                points = points[-2:] + points[:2]  # 调整为左上顺时针顺序

            except ValueError as e:
                raise ValueError(f"图像文件名格式错误: {e}")

            # 创建结果文件夹
            result_folder = os.path.join(self.folder_path, 'results')
            os.makedirs(result_folder, exist_ok=True)

            # 处理图像并保存结果
            output_path = os.path.join(result_folder, f'result_{current_image}')
            plate_path = os.path.join(result_folder, f'plate_{current_image}')

            # 提取车牌号
            label = class1_rcg.extract_license_plate_label(label_info)

            # 绘制原始图像的标注
            class1_rcg.ImgShow(imgpath, box, points, label, output_path)

            # 提取车牌区域
            extracted_plate = class1_rcg.extract_license_plate_image(imgpath, points, plate_path)
            # 添加：将提取的车牌图像保存到CNN验证目录
            cnn_verify_path = "../2_verify_with_cnn/result/yes"
            os.makedirs(cnn_verify_path, exist_ok=True)
            cnn_plate_path = os.path.join(cnn_verify_path, f'plate_{current_image}')
            if os.path.exists(plate_path):
                import shutil
                shutil.copy2(plate_path, cnn_plate_path)

            # 对车牌图像进行分割
            img = cv2.imread(cnn_plate_path)
            img = cv2.resize(img, (250, 70))
            # 转为灰度图
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 调整亮度
            mean_brigthness = np.mean(gray_img)
            if mean_brigthness > 127:
                gray_img = cv2.convertScaleAbs(gray_img, alpha=1, beta=-50)
            else:
                gray_img = cv2.convertScaleAbs(gray_img, alpha=1, beta=50)

            # 去除边缘并处理
            new_img = split_single.removeBorder(gray_img)
            kernel = np.ones((1, 1), np.uint8)
            new_img = cv2.dilate(new_img, kernel, iterations=1)
            # 创建分割图像的保存目录
            split_folder = f"../3_split/outcut/{self.current_image_index}"
            os.makedirs(os.path.dirname(split_folder), exist_ok=True)
            # 进行字符分割
            if split_single.img_separate(new_img, "../3_split/outcut", str(self.current_image_index)):
                # 显示分割后的字符图像
                image_files = [f for f in os.listdir(split_folder)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                if image_files:
                    self.display_combined_images(split_folder, sorted(image_files))
                # 添加：使用模型进行识别
                # 执行识别流程，与II类车牌相同
                data = straight_model1.main()
                label_predicted = data[self.current_image_index]  # 模型预测的结果

                # 从文件名中获取真实标签
                def generate_label_from_filename(filename, provincelist, wordlist):
                    try:
                        _, _, _, _, label_info, _, _ = filename.split('-')
                        label_parts = label_info.split('_')
                        province = provincelist[int(label_parts[0])]
                        words = [wordlist[int(i)] for i in label_parts[1:]]
                        return province + ''.join(words)
                    except Exception as e:
                        raise ValueError(f"Failed to generate label from filename: {e}")

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
                # 获取真实标签
                label_true = generate_label_from_filename(current_image, provincelist, wordlist)

                # 比较预测结果和真实标签
                is_correct = label_predicted == label_true
                if is_correct:
                    self.correct_count += 1
                self.total_count += 1

                # 生成字符对比文本
                comparison_text = "字符对比：\n"
                for i, (char1, char2) in enumerate(zip(label_predicted, label_true)):
                    match_status = "✓" if char1 == char2 else "✗"
                    comparison_text += f"位置{i + 1}: {char1} vs {char2} {match_status}\n"

                # 计算正确率
                accuracy = (self.correct_count / self.total_count * 100) if self.total_count > 0 else 0

                # 更新显示
                self.comparison_label.config(text=comparison_text)
                self.accuracy_label.config(text=f"正确率：{accuracy:.2f}% ({self.correct_count}/{self.total_count})")

                # 显示处理后的图像和预测结果
                self.display_processed_images(output_path, plate_path, label_predicted)

            # 更新处理状态
            self.info_label.config(text="处理完成！")
            return True

        except Exception as e:
            self.info_label.config(text=f"处理失败: {str(e)}")
            return False

    def process_class2_image(self):
        """处理II类车牌图像"""
        try:
            current_image = self.image_files[self.current_image_index]
            # 检查缓存中是否已有处理结果
            if (current_image in self.cache['processed_images'] and 
                self.current_process_type == "II类车牌集中测试"):
                self.display_cached_results(current_image)
                return True# 第一次处理时执行批量处理

            # 如果没有处理线程在运行，启动新的处理线程
            if not self.processing_thread or not self.processing_thread.is_alive():
                self.info_label.config(text="正在处理中...")
                self.processing_thread = threading.Thread(
                    target=self._process_images_thread,
                    args=(current_image,)
                )
                self.processing_thread.daemon = True
                self.processing_thread.start()
            
                # 启动定期检查结果的函数
                self.root.after(100, self._check_processing_results)

            return True

        except Exception as e:
            self.info_label.config(text=f"处理失败: {str(e)}")
            return False

    def _process_images_thread(self, current_image):
        """后台处理图像的线程函数"""
        try:
            # 批量处理
            if not hasattr(self, '_batch_processed'):
                Coarse_positioning.main(self.folder_path)
                recognize_VL.main()
                split_single.main()
            
                # 根据选择的模型执行不同的识别过程
                selected_model = self.model_var.get()
                if selected_model == "L1":
                    self.recognition_data = recognize_l.main()
                    self.info_label.config(text="使用L1模型进行识别...")
                else:  # L2 或其他情况
                    self.recognition_data = straight_model1.main()
                    self.info_label.config(text="使用L2模型进行识别...")
            
                self._batch_processed = True

            # 处理当前图像
            result_image_path = os.path.join(self.folder_path, current_image)
            plate_image_path = os.path.join('1_cu_place/cu_result', f'extracted_{current_image}')
            label = self.recognition_data[self.current_image_index]

            # 获取真实标签
            label_true = self.generate_label_from_filename(current_image)

            # 比较预测结果和真实标签
            is_correct = label == label_true
            if is_correct:
                self.correct_count += 1
            self.total_count += 1

            # 生成字符对比文本
            comparison_text = "字符对比：\n"
            for i, (char1, char2) in enumerate(zip(label, label_true)):
                match_status = "✓" if char1 == char2 else "✗"
                comparison_text += f"位置{i + 1}: {char1} vs {char2} {match_status}\n"

            # 计算正确率
            accuracy = (self.correct_count / self.total_count * 100) if self.total_count > 0 else 0

            # 获取字符分割图像
            base_folder = "3_split/outcut"
            subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
            subfolders.sort(key=lambda x: int(os.path.basename(x)))
            all_image = subfolders[self.current_image_index]
            image_files = [f for f in os.listdir(all_image) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            # 将结果放入队列
            self.result_queue.put({
                'result_path': result_image_path,
                'plate_path': plate_image_path,
                'label': label,
                'label_true': label_true,
                'comparison_text': comparison_text,
                'accuracy': accuracy,
                'folder': all_image,
                'files': image_files,
                'current_image': current_image
            })

        except Exception as e:
            self.result_queue.put({'error': str(e)})

    def generate_label_from_filename(self, filename):
        """从文件名生成真实标签"""
        try:
            provincelist = [
                "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
                "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                "桂", "琼", "川", "贵", "云", "西", "陕", "甘", "青", "宁", "新"
            ]
            wordlist = [
                "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
                "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
                "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5",
                "6", "7", "8", "9"
            ]
        
            # 根据图像名分割标注
            _, _, _, _, label_info, _, _ = filename.split('-')
            label_parts = label_info.split('_')
        
            # 生成车牌号
            province = provincelist[int(label_parts[0])]
            words = [wordlist[int(i)] for i in label_parts[1:]]
            return province + ''.join(words)
        
        except Exception as e:
            raise ValueError(f"Failed to generate label from filename: {e}")

    def _check_processing_results(self):
        """检查处理结果并更新界面"""
        try:
            if not self.result_queue.empty():
                result = self.result_queue.get_nowait()
            
                if 'error' in result:
                    self.info_label.config(text=f"处理失败: {result['error']}")
                    return
                
                # 显示结果
                self.display_processed_images(
                    result['result_path'],
                    result['plate_path'],
                    result['label']
                )
                self.display_combined_images(
                    result['folder'],
                    result['files']
                )
            
                # 更新字符对比和正确率显示
                self.comparison_label.config(text=result['comparison_text'])
                self.accuracy_label.config(text=f"正确率：{result['accuracy']:.2f}% ({self.correct_count}/{self.total_count})")
            
                # 更新缓存
                current_image = result['current_image']
                self.cache['processed_images'][current_image] = result['result_path']
                self.cache['plate_images'][current_image] = result['plate_path']
                self.cache['plate_numbers'][current_image] = result['label']
                self.cache['combined_images'][current_image] = {
                    'folder': result['folder'],
                    'files': result['files']
                }
            
                self.info_label.config(text="处理完成！")
            
            # 如果处理线程还在运行，继续检查
            if self.processing_thread and self.processing_thread.is_alive():
                self.root.after(100, self._check_processing_results)
            
        except Exception as e:
            self.info_label.config(text=f"更新结果失败: {str(e)}")

        

    def display_cached_results(self, image_name):
        """显示缓存的处理结果"""
        try:
            # 显示处理后的图像和车牌区域
            self.display_processed_images(
                self.cache['processed_images'][image_name],
                self.cache['plate_images'][image_name],
                self.cache['plate_numbers'][image_name]
            )
            
            # 如果是II类车牌，显示字符分割结果
            if (self.current_process_type == "II类车牌集中测试" and 
                image_name in self.cache['combined_images']):
                combined_data = self.cache['combined_images'][image_name]
                self.display_combined_images(
                    combined_data['folder'],
                    combined_data['files']
                )
                
            self.info_label.config(text="显示缓存结果")
            
        except Exception as e:
            self.info_label.config(text=f"显示缓存结果失败: {str(e)}")

    def display_combined_images(self, folder_path, image_files):
        """拼接所有图片并在界面上显示"""
        try:
            images = []

            # 加载所有图片
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                img = Image.open(image_path)
                img.thumbnail((200, 60))  # 缩小图片到统一大小
                images.append(img)

            # 拼接所有图片
            widths, heights = zip(*(img.size for img in images))
            total_width = sum(widths)
            max_height = max(heights)

            combined_image = Image.new("RGB", (total_width, max_height))

            x_offset = 0
            for img in images:
                combined_image.paste(img, (x_offset, 0))
                x_offset += img.width

            # 显示拼接后的图像
            combined_photo = ImageTk.PhotoImage(combined_image)
            self.combined_label.config(image=combined_photo, text="")
            self.combined_label.image = combined_photo

        except Exception as e:
            self.info_label.config(text=f"拼接图片失败: {str(e)}")

    def display_processed_images(self, result_path, plate_path, label):
        """显示处理后的图像结果"""
        try:
            # 显示标注后的原始图像
            result_image = Image.open(result_path)
            result_image.thumbnail((450, 450))
            result_photo = ImageTk.PhotoImage(result_image)
            self.image_label.config(image=result_photo)
            self.image_label.image = result_photo

            # 显示提取的车牌区域
            plate_image = Image.open(plate_path)
            plate_image = plate_image.resize((400, 120))
            plate_photo = ImageTk.PhotoImage(plate_image)
            self.plate_image_label.config(image=plate_photo)
            self.plate_image_label.image = plate_photo

            # 显示识别结果
            self.plate_text_label.config(text=f"车牌号码：{label}")

            # 保存识别结果
            result_text = (
                f"图像文件：{os.path.basename(result_path)}\n"
                f"车牌号码：{label}\n"
                f"处理时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            result_file = os.path.join(os.path.dirname(result_path), "recognition_results.txt")
            with open(result_file, "a", encoding="utf-8") as f:
                f.write(result_text + "\n\n")

        except Exception as e:
            raise Exception(f"显示处理结果失败: {str(e)}")

    def start_recognition(self):
        """开始识别处理"""
        selected_option = self.option_var.get()

        if not self.image_files:
            self.info_label.config(text="请先加载图像文件！")
            return


        # 如果切换了处理类型，清除缓存和批量处理标记
        if self.current_process_type != selected_option:
            self.cache = {
                'processed_images': {},
                'plate_images': {},
                'combined_images': {},
                'plate_numbers': {},
            }
            if hasattr(self, '_batch_processed'):
                delattr(self, '_batch_processed')
    
        self.current_process_type = selected_option
        

        if selected_option == "I类车牌集中测试":
            try:
                if self.process_class1_image():
                    self.update_navigation_buttons()
                    # 记录当前处理类型
                    self.current_process_type = "I类车牌集中测试"
            except Exception as e:
                self.info_label.config(text=f"处理失败: {str(e)}")

        elif selected_option == "II类车牌集中测试":
            try:
                if self.process_class2_image():
                    self.update_navigation_buttons()
            except Exception as e:
                self.info_label.config(text=f"处理失败: {str(e)}")
        else:
            self.info_label.config(text="请选择处理类型")

def main():
    root = tk.Tk()
    app = LicensePlateRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()