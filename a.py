import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

class FaceDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("人脸特征检测系统")
        
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
        
    def create_widgets(self):
        # 视频显示区域
        self.video_label = tk.Label(self.window)
        self.video_label.pack(padx=10, pady=10)
        
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
            print("无法打开摄像头")
            return
        
        self.start_btn.config(state=tk.DISABLED)
        self.detect_btn.config(state=tk.NORMAL)
        
        # 启动视频流线程
        self.video_thread = threading.Thread(target=self.update_video)
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def toggle_detection(self):
        self.detecting = not self.detecting
        self.detect_btn.config(text="停止识别" if self.detecting else "识别")
    
    def detect_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 人脸检测
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # 眼睛检测
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # 鼻子检测
            noses = self.nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
            for (nx, ny, nw, nh) in noses:
                cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (0, 0, 255), 2)
            
            # 嘴巴检测
            mouths = self.mouth_cascade.detectMultiScale(roi_gray, 1.3, 5)
            for (mx, my, mw, mh) in mouths:
                cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (255, 0, 255), 2)
        
        return frame
    
    def update_video(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if self.detecting:
                frame = self.detect_features(frame)
            
            # 转换为RGB格式并在GUI显示
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        if self.cap:
            self.cap.release()
    
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