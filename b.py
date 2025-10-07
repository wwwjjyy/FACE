import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import os
import sys
import platform

class FaceDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("人脸特征检测系统")
        self.window.geometry("800x600")
        
        # 加载Haar级联分类器
        self.face_cascade = cv2.CascadeClassifier('.\haar\haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('.\haar\haarcascade_eye.xml')
        self.nose_cascade = cv2.CascadeClassifier('.\haar\haarcascade_mcs_nose.xml')
        self.mouth_cascade = cv2.CascadeClassifier('.\haar\haarcascade_mcs_mouth.xml')
        
        # 创建GUI组件
        self.create_widgets()
        
        # 初始化摄像头状态
        self.cap = None
        self.detecting = False
        self.running = True
        
        # 无特征检测计时相关变量
        self.last_feature_time = None
        self.shutdown_timer = None
        self.shutdown_window = None
        self.shutdown_countdown = 30
        self.shutdown_in_progress = False
        self.no_feature_timer = None
        self.no_feature_countdown = 10
        
        # 状态标签
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_label = tk.Label(self.window, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_widgets(self):
        # 视频显示区域
        self.video_label = tk.Label(self.window)
        self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # 按钮区域
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(pady=10)
        
        self.start_btn = ttk.Button(btn_frame, text="启动摄像头", command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.detect_btn = ttk.Button(btn_frame, text="识别", command=self.toggle_detection, state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT, padx=5)
        
        self.exit_btn = ttk.Button(btn_frame, text="退出", command=self.exit_app)
        self.exit_btn.pack(side=tk.LEFT, padx=5)
    
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头")
            return
        
        self.start_btn.config(state=tk.DISABLED)
        self.detect_btn.config(state=tk.NORMAL)
        self.status_var.set("摄像头已启动")
        
        # 启动视频流线程
        self.video_thread = threading.Thread(target=self.update_video)
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def toggle_detection(self):
        self.detecting = not self.detecting
        self.detect_btn.config(text="停止识别" if self.detecting else "识别")
        self.status_var.set("检测中..." if self.detecting else "检测已停止")
        
        # 重置无特征检测计时
        if self.detecting:
            self.last_feature_time = time.time()
            self.cancel_shutdown()
    
    def detect_features(self, frame):
        """检测人脸特征并返回是否检测到任何特征"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        feature_detected = False
        
        # 人脸检测
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            feature_detected = True
            
            # 眼睛检测
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                feature_detected = True
            
            # 鼻子检测
            noses = self.nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
            for (nx, ny, nw, nh) in noses:
                cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (0, 0, 255), 2)
                feature_detected = True
            
            # 嘴巴检测
            mouths = self.mouth_cascade.detectMultiScale(roi_gray, 1.3, 5)
            for (mx, my, mw, mh) in mouths:
                cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (255, 0, 255), 2)
                feature_detected = True
        
        return frame, feature_detected
    
    def update_no_feature_timer(self):
        """更新无特征检测计时"""
        if not self.detecting or self.shutdown_in_progress:
            return
            
        current_time = time.time()
        if self.last_feature_time is None:
            self.last_feature_time = current_time
            return
            
        # 计算无特征持续时间
        idle_time = current_time - self.last_feature_time
        remaining = max(0, self.no_feature_countdown - int(idle_time))
        
        # 更新状态栏显示
        self.status_var.set(f"无特征检测中: 将在{remaining}秒后触发关机倒计时")
        
        # 如果超过60秒，显示关机倒计时窗口
        if idle_time >= self.no_feature_countdown and not self.shutdown_window:
            self.show_shutdown_window()
        
        # 继续计时
        if self.running and self.detecting:
            self.no_feature_timer = self.window.after(1000, self.update_no_feature_timer)
    
    def show_shutdown_window(self):
        """显示关机倒计时窗口"""
        if self.shutdown_window:
            return
            
        self.shutdown_in_progress = True
        self.shutdown_countdown = 30
        
        # 创建关机倒计时窗口
        self.shutdown_window = tk.Toplevel(self.window)
        self.shutdown_window.title("系统即将关机")
        self.shutdown_window.geometry("400x200")
        self.shutdown_window.resizable(False, False)
        self.shutdown_window.protocol("WM_DELETE_WINDOW", self.cancel_shutdown)
        self.shutdown_window.attributes('-topmost', True)  # 保持在最前面
        
        # 倒计时标签
        self.shutdown_label_var = tk.StringVar()
        self.shutdown_label_var.set(f"系统将在 {self.shutdown_countdown} 秒后关机")
        shutdown_label = tk.Label(self.shutdown_window, textvariable=self.shutdown_label_var, font=("Arial", 16))
        shutdown_label.pack(pady=20)
        
        # 信息标签
        info_label = tk.Label(self.shutdown_window, text="检测到长时间无人使用，系统将自动关机", font=("Arial", 12))
        info_label.pack(pady=10)
        
        # 按钮区域
        btn_frame = tk.Frame(self.shutdown_window)
        btn_frame.pack(pady=15)
        
        shutdown_btn = ttk.Button(btn_frame, text="立即关机", command=self.shutdown_now)
        shutdown_btn.pack(side=tk.LEFT, padx=10)
        
        cancel_btn = ttk.Button(btn_frame, text="取消关机", command=self.cancel_shutdown)
        cancel_btn.pack(side=tk.LEFT, padx=10)
        
        # 启动倒计时
        self.update_shutdown_countdown()
    
    def update_shutdown_countdown(self):
        """更新关机倒计时"""
        if not self.shutdown_window or self.shutdown_countdown <= 0:
            return
            
        self.shutdown_countdown -= 1
        self.shutdown_label_var.set(f"系统将在 {self.shutdown_countdown} 秒后关机")
        
        # 倒计时结束后关机
        if self.shutdown_countdown <= 0:
            self.shutdown_now()
            return
            
        # 每秒更新一次
        self.shutdown_timer = self.window.after(1000, self.update_shutdown_countdown)
    
    def cancel_shutdown(self):
        """取消关机操作"""
        if self.shutdown_timer:
            self.window.after_cancel(self.shutdown_timer)
            self.shutdown_timer = None
            
        if self.no_feature_timer:
            self.window.after_cancel(self.no_feature_timer)
            self.no_feature_timer = None
            
        if self.shutdown_window:
            self.shutdown_window.destroy()
            self.shutdown_window = None
            
        self.shutdown_in_progress = False
        self.last_feature_time = time.time()  # 重置计时
        self.status_var.set("检测已恢复")
        
        # 重启无特征检测计时
        if self.detecting:
            self.no_feature_timer = self.window.after(1000, self.update_no_feature_timer)
    
    def shutdown_now(self):
        """立即关机"""
        self.running = False
        
        # 关闭摄像头
        if self.cap:
            self.cap.release()
        
        # 关闭窗口
        if self.shutdown_window:
            self.shutdown_window.destroy()
        self.window.destroy()
        
        # 执行关机命令
        try:
            if platform.system() == "Windows":
                os.system("shutdown /s /t 0")
            elif platform.system() == "Darwin":  # macOS
                os.system("sudo shutdown -h now")
            else:  # Linux
                os.system("sudo shutdown -h now")
        except:
            messagebox.showinfo("关机", "系统即将关机")
            sys.exit(0)
    
    def update_video(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.status_var.set("摄像头读取错误")
                break
            
            if self.detecting:
                frame, feature_detected = self.detect_features(frame)
                
                # 更新特征检测时间
                if feature_detected:
                    self.last_feature_time = time.time()
                    if self.shutdown_in_progress:
                        self.cancel_shutdown()
                    elif self.no_feature_timer is None:
                        # 启动无特征检测计时
                        self.no_feature_timer = self.window.after(1000, self.update_no_feature_timer)
            
            # 转换为RGB格式并在GUI显示
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img.thumbnail((800, 600))  # 调整图像大小以适应窗口
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def exit_app(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.exit_app)
    root.mainloop()