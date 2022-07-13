import os
import threading
from ctypes import cdll

import cv2
import sys
import time
import datetime
import json

import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from ui.ui import *
from ui.login import *
import subprocess
import shutil
import pyttsx3
import io

# import sys
# sudo apt-get install espeak
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 加载人脸识别参数文件
face_cascade = cv2.CascadeClassifier("./data/haarcascade_frontaLface_default.xml")


# 中文语音播报
def say_zh(msg):
    engine = pyttsx3.init()
    engine.setProperty('voice', 'zh')
    engine.setProperty('rate', 180)
    engine.say(msg)
    engine.runAndWait()


chinese_num_dict = {'0': '零', '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
                    '.': '点'}


# 数字转换汉字
def to_zh(num):
    num_int = int(num * 100)
    chinese_num = ""
    num_int = str(num_int)
    if len(num_int) == 4:
        if num_int[0] != '1':
            chinese_num = chinese_num_dict[num_int[0]] + "十" + chinese_num_dict[num_int[1]] + "点" + chinese_num_dict[
                num_int[2]] + chinese_num_dict[num_int[3]] + "度"
        else:
            chinese_num = "十" + chinese_num_dict[num_int[1]] + "点" + chinese_num_dict[num_int[2]] + chinese_num_dict[
                num_int[3]] + "度"
    elif len(num_int) == 3:
        chinese_num = chinese_num_dict[num_int[0]] + "点" + chinese_num_dict[num_int[1]] + chinese_num_dict[
            num_int[2]] + "度"
    else:
        return "温度测量错误"
    return "体温" + chinese_num


class State(object):
    def __init__(self):
        # 初始化开始时间
        self.init_time = time.time()
        # 多久抽取一张照片识别
        self.interval = 2
        # 前一个人名字
        self.before = None
        # 现在识别到的名字
        self.after_name = None
        # 距离、测量间隔
        self.distance = 1000
        self.temperature = 0


class Horn_Sensor(object):
    def __init__(self):
        self.init_time = time.time()
        self.flag = False


user_state = State()
user_horn_sensor = Horn_Sensor()


# 初始化全局变量
class GlobalVariable():
    # 参数保存字典
    global_var = {}

    # 线程锁确保多线程读取安全
    # lock = threading.RLock()
    # 初始化参数
    def init(self):

        now_time = datetime.datetime.now()
        self.global_var = {
            'login_Admin': False,
            'login_Usr': False,
            'login_time': now_time,
            'Usr_name': 'unkonw',
            'label_isWrok': False,
        }

    # 设置参数
    def set_var(self, name, value):
        # 加锁
        # self.lock.acquire()
        try:
            self.global_var[name] = value
        finally:
            # 释放
            # self.lock.release()
            pass

    # 获取参数
    def get_var(self, name):

        try:  # 加锁
            # self.lock.acquire()
            return self.global_var[name]
        finally:
            # 释放
            # self.lock.release()
            pass


global_variable = GlobalVariable()
global_variable.init()


# 打开摄像头
class WorkThread1(QThread):
    signals = pyqtSignal(object)  # 定义信号对象,传递值为str类型，使用int，可以为int类型

    def __init__(self):  # 向线程中传递参数，以便在run方法中使用
        super(WorkThread1, self).__init__()
        self.isWork = True

    def run(self):  # 重写run方法
        cam = cv2.VideoCapture(0)
        while cam.isOpened() and self.isWork:
            try:
                ret, img = cam.read()
                if ret:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(img)

                    if self.isWork:
                        self.signals.emit(pixmap)  # 发射信号，str类型数据，内容为需要传递的数据
                    else:
                        cam.release()
            except:
                continue

    def stop(self):
        self.isWork = False


# 人脸识别
class WorkThread2(QThread):
    signals = pyqtSignal(object)  # 定义信号对象,传递值为str类型，使用int，可以为int类型

    def __init__(self, face_model_filename):  # 向线程中传递参数，以便在run方法中使用
        super(WorkThread2, self).__init__()
        self.isWork = True
        self.face_model_fileName = face_model_filename

    def run(self):  # 重写run方法
        names = os.listdir('./dataset')
        # 创建识别模型，使用EigenFace算法识别，Confidence评分低于4000是可靠
        # model = cv2.face.EigenFaceRecognizer_create()
        # 创建识别模型，使用LBPHFace算法识别，Confidence评分低于50是可靠
        # model = cv2.face.LBPHFaceRecognizer_create()
        # 创建识别模型，使用FisherFace算法识别，Confidence评分低于4000是可靠
        model = cv2.face.FisherFaceRecognizer_create()
        # 加载模型参数
        model.read(self.face_model_fileName)
        # 打开本地摄像头
        cam = cv2.VideoCapture(0)
        # 加载Haar级联数据文件，用于检测人面
        face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

        while cam.isOpened() and self.isWork:
            # 检测摄像头的人面

            try:
                ret, img = cam.read()

                if ret:
                    faces = face_cascade.detectMultiScale(img, 1.3, 5)
                    # 将检测的人面进行识别处理
                    for (x, y, w, h) in faces:
                        # 画出人面所在位置并灰度处理
                        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        roi = gray[x:x + w, y:y + h]
                        # 将检测的人面缩放200*200大小，用于识别
                        # cv2.INTER_LINEAR是图片变换方式，其余变换方式如下：
                        # INTER_NN - 最近邻插值。
                        # INTER_LINEAR - 双线性插值(缺省使用)
                        # INTER_AREA - 使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现。
                        # INTER_CUBIC - 立方插值。
                        roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                        # 检测的人面与模型进行匹配识别
                        params = model.predict(roi)
                        # print("Label: %s, Confidence: %.2f" % (params[0], params[1]))
                        # 将识别结果显示在摄像头上
                        # cv2.FONT_HERSHEY_SIMPLEX 定义字体
                        # cv2.putText参数含义：图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
                        # 如果要输出中文字，可参考https://blog.csdn.net/m0_37606112/article/details/78511381
                        score = params[1] / 4000
                        if score > 1:
                            score = 1
                        if score > 0.6:
                            text = names[params[0]] + ' score:%.3f' % (score)

                            # print("text= ", text)
                        else:
                            text = 'UnKnow'
                        cv2.putText(img, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(img)

                    if self.isWork:
                        self.signals.emit(pixmap)  # 发射信号，str类型数据，内容为需要传递的数据
                    else:
                        cam.release()
            except:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(img)
                if self.isWork:
                    self.signals.emit(pixmap)  # 发射信号，str类型数据，内容为需要传递的数据
                else:
                    cam.release()
                continue

    def stop(self):
        self.isWork = False


# 录制人脸数据
class WorkThread3(QThread):
    signals_img = pyqtSignal(object)  # 定义信号对象,传递图片,
    signals_result = pyqtSignal(str)  # 定义信号对象,传递处理结果,

    def __init__(self, face_data_name):  # 向线程中传递参数，以便在run方法中使用
        super(WorkThread3, self).__init__()
        self.isWork = True
        self.startTime = datetime.datetime.now()
        self.seconds = 0
        self.face_data_name = face_data_name
        if not os.path.exists('./dataset/' + face_data_name):
            os.mkdir('./dataset/' + face_data_name)

    def run(self):  # 重写run方法
        # 打开本地摄像头
        cam = cv2.VideoCapture(0)
        # 加载Haar级联数据文件，用于检测人面
        face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        count = 0

        init_time = time.time()

        while cam.isOpened() and self.isWork:

            # 倒数5秒开始录制人脸
            if self.seconds <= 5:
                end_time = datetime.datetime.now()
                self.seconds = (end_time - self.startTime).seconds
            # 保存20张后退出
            if count >= 20:
                self.signals_img.emit('')
                self.signals_result.emit('model done')
                break
            try:
                ret, img = cam.read()
                if ret:
                    # 转换成灰度图
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # ROI区域
                    x_min = int(img.shape[1] * 0.2)
                    y_min = int(img.shape[0] * 0.15)
                    x_max = int(img.shape[1] * 0.8)
                    y_max = int(img.shape[0] * 0.85)
                    # 获取圆的圆心和半径
                    cir_center = (int(img.shape[1] / 2), int(img.shape[0] / 2))  # (h, w) -> (x, y)
                    cir_radius = int(img.shape[0] / 2.5)
                    # 画圆形ROI区域
                    img = cv2.circle(img, cir_center, cir_radius, (0, 255, 0), 3)
                    # 查找人脸
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    if len(faces) > 0:
                        # 取第一个结果
                        x, y, w, h = faces[0]
                        if x > x_min and y > y_min and x < x_max and y < y_max:
                            # 画出面部位置
                            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            # 根据人脸的位置截取图片并调整截取后的图片大小
                            gray_roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200), interpolation=cv2.INTER_LINEAR)

                            if self.seconds > 5:

                                if time.time() - init_time > 0.6:
                                    # 保存图片
                                    count += 1
                                    self.signals_result.emit('保存第{}张图片，还剩{}张'.format(count, 20 - count))
                                    save_path = os.path.join('dataset', self.face_data_name, str(count) + '.jpg')
                                    cv2.imwrite(save_path, gray_roi)
                                    init_time = time.time()

                            else:
                                self.signals_result.emit('{}秒后开始录入'.format(5 - self.seconds))
                        else:
                            self.signals_result.emit('超出范围！')
                    else:
                        self.signals_result.emit('未检测到人脸！')

                    # 将图片转成QImage,发送给界面显示
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(img)

                    if self.isWork:
                        self.signals_img.emit(pixmap)  # 发射信号，str类型数据，内容为需要传递的数据
                    else:
                        self.signals_img.emit('')
                        self.signals_result.emit('')

            except:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(img)
                if self.isWork:
                    self.signals_img.emit(pixmap)  # 发射信号，str类型数据，内容为需要传递的数据
                else:
                    cam.release()
                continue

    def stop(self):
        self.isWork = False


# 训练人脸数据
class WorkThread4(QThread):
    signals_result = pyqtSignal(str)

    def __init__(self):  # 向线程中传递参数，以便在run方法中使用
        super(WorkThread4, self).__init__()
        self.isWork = True

    def run(self):  # 重写run方法
        # 获取人脸名字，以文件夹命名，无用的文件夹要删除掉！
        names = os.listdir('./dataset')
        [X, y] = self.read_images('./dataset')
        # 创建识别模型，使用EigenFace算法识别，Confidence评分低于4000是可靠
        # model = cv2.face.EigenFaceRecognizer_create()
        # 创建识别模型，使用LBPHFace算法识别，Confidence评分低于50是可靠
        # model = cv2.face.LBPHFaceRecognizer_create()
        # 创建识别模型，使用FisherFace算法识别，Confidence评分低于4000是可靠
        model = cv2.face.FisherFaceRecognizer_create()
        # # 训练模型
        # # train函数参数：images, labels，两参数必须为np.array格式，而且labels的值必须为整型
        self.signals_result.emit('模型训练中，请勿操作')
        model.train(np.array(X), np.array(y))
        # 保存模型
        model.save("./model/model.xml")
        self.signals_result.emit('model done')

        self.signals_result.emit("模型训练完成")

    # 加载人脸图片
    def read_images(self, path, sz=None):
        """Reads the images in a given folder, resizes images on the fly if size is given.
        Args:
            path: 人面数据所在的文件路径
            sz: 图片尺寸设置
        Returns:
            A list [X,y]
                X: 图片信息
                y: 图片的读取顺序
        """
        c = 0
        x, y = [], []

        for dirname, dirnames, filenames in os.walk(path):
            for subdirname in dirnames:
                subject_path = os.path.join(dirname, subdirname)
                for filename in os.listdir(subject_path):
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    x.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                c = c + 1

        return [x, y]


# 自动运行
class AutoWorkTread(QThread):
    signals_img = pyqtSignal(object)  # 定义信号对象,传递图片,
    signals_result = pyqtSignal(str)  # 定义信号对象,传递识别结果,

    def __init__(self):  # 向线程中传递参数，以便在run方法中使用
        super(AutoWorkTread, self).__init__()
        self.isWork = True
        self.wait_time = 100  # 扫描人脸等待时间
        self.names = os.listdir('./dataset')
        self.startTime = datetime.datetime.now()
        self.dll = cdll.LoadLibrary('app/SensorControl.so')

    def run(self):  # 重写run方法
        # 设定相机
        cam = cv2.VideoCapture(0)
        # 人脸识别
        face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        # 人脸匹配
        model = cv2.face.FisherFaceRecognizer_create()
        # 加载模型参数
        with open('data/usrdata.json', 'r') as f:
            data = json.load(f)
        face_model_filename = data['Setup']['model_path']
        model.read(face_model_filename)

        # 设置有识别人脸时间
        start_time_1 = time.time()
        # 初始化传感器
        self.dll.Sensor_init()
        while cam.isOpened() and self.isWork:
            try:
                ret, img = cam.read()
                if ret:
                    # 转换成灰度图
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # ROI区域
                    x_min = int(img.shape[1] * 0.2)
                    y_min = int(img.shape[0] * 0.15)
                    x_max = int(img.shape[1] * 0.8)
                    y_max = int(img.shape[0] * 0.85)
                    # gray_roi = gray[x_min:x_max, y_min:y_max]
                    # 显示ROI区域
                    # img = cv2.rectangle(img,(x_min,y_min),(x_max,y_max),(255, 0, 0), 2)
                    # 获取圆的圆心和半径
                    cir_center = (int(img.shape[1] / 2), int(img.shape[0] / 2))  # (h, w) -> (x, y)
                    cir_radius = int(img.shape[0] / 2.5)
                    # 画圆形ROI区域
                    img = cv2.circle(img, cir_center, cir_radius, (0, 255, 0), 3)

                    user_info = ['超出检测范围！', '无法识别身份！', '距离太近，请站远点！', '距离太远，请靠近点！', None]

                    # ************************************************************************************

                    def identify_face():

                        self.dll.Sensor_Control(3)
                        # 超声波距离
                        distance = self.dll.Sensor_Control(1) / 100
                        # 体温
                        user_state.temperature = self.dll.Sensor_Control(4) / 100

                        try:
                            # 查找人脸
                            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                            if len(faces) > 0:
                                x, y, w, h = faces[0]
                                if x < x_min or y < y_min or x + w > x_max or y + h > y_max:
                                    user_state.init_time = time.time()
                                    user_state.after_name = '超出检测范围！'
                                else:
                                    user_state.init_time = time.time()
                                    gray_roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200),
                                                          interpolation=cv2.INTER_LINEAR)
                                    # 人脸匹配，输出得分结果
                                    params = model.predict(gray_roi)
                                    score = params[1] / 4000
                                    if score > 1:
                                        score = 1
                                    if score > 0.6:
                                        user_state.after_name = self.names[params[0]] + " " + str(
                                            user_state.temperature) + "度"
                                    else:
                                        user_state.after_name = '无法识别身份！'

                                if distance < 10:
                                    user_state.after_name = '距离太近，请站远点！'
                                elif distance > 120:
                                    user_state.after_name = '距离太远，请靠近点！'

                            else:
                                user_state.after_name = None
                        except KeyError:
                            user_state.after_name = None

                    # ************************************************************************************

                    if time.time() - start_time_1 > 4:
                        start_time_1 = time.time()
                        identify_face()

                        if user_state.after_name not in user_info:
                            say_str = to_zh(user_state.temperature)
                            t1 = threading.Thread(target=say_zh, args=(say_str,))
                            t1.start()

                            if user_state.temperature < 37.2:
                                self.dll.Sensor_Control(2)

                    if time.time() - user_state.init_time > 30:
                        self.signals_img.emit('')  # 发射信号
                        self.signals_result.emit('')
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(img)
                        if self.isWork:
                            self.signals_img.emit(pixmap)  # 发射信号
                            self.signals_result.emit(user_state.after_name)

            except:
                continue
        # 退出关闭相机
        cam.release()

    # 停止函数
    def stop(self):
        self.isWork = False


# 登录界面
class MyLogin(QDialog, Ui_Dialog):
    def __init__(self):
        super(MyLogin, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('用户登录')


# 注册界面
class MyRegister(QDialog, Ui_Dialog):
    def __init__(self):
        super(MyRegister, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('用户注册')


# 注销界面
class MySignOut(QDialog, Ui_Dialog):
    def __init__(self):
        super(MySignOut, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('用户注销')
        self.lineEdit_2.setVisible(False)

        with open('data/usrdata.json', 'r') as f:
            data = json.load(f)
        text = ''
        usr_count = 0
        for usr in data['Usr'].keys():
            usr_count += 1
            text += ' {' + usr + '} '
        text = '共有{}个用户'.format(usr_count) + text
        self.label_2.setText('用户列表')
        self.label_2.setStyleSheet('background: yellow')
        self.label_2.setToolTip(text)


# 主界面
class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        '''菜单栏按键事件'''
        # 登录
        self.Login.triggered.connect(self.open_login_form)
        # 注册
        self.Register.triggered.connect(self.open_register_form)
        # 注销
        self.Signout.triggered.connect(self.open_signout_form)
        # 关闭程序
        self.CloseWindow.triggered.connect(self.closeEvent)
        # 打开自动运行
        self.AutoRun.triggered.connect(lambda: self.AotuRun_thread_start())
        # 打开摄像头
        self.Open_cam.triggered.connect(lambda: self.Open_cam_fc_start_thread())
        # 打开人脸识别
        self.Open_face_regionizer.triggered.connect(lambda: self.Open_face_regionizer_fc_start_thread())
        # 录制人脸数据
        self.Face_data_get.triggered.connect(lambda: self.Face_data_get_fc_start_thread())
        # 选择模型文件

        # self.Select_model_file.triggered.connect(self.Select_model_file_fc)
        # 开始训练
        self.Train_face_data.triggered.connect(self.Train_face_data_fc_start_thread)

        # 时间日期
        self.timer = QTimer()
        self.timer.start()
        self.timer.timeout.connect(self.clock)
        # 开机启动
        self.AutoRun.setText('停止运行')
        self.AotuRun_thread = AutoWorkTread()  # 类的实例化
        self.AotuRun_thread.start()  # 开启线程
        self.AotuRun_thread.signals_img.connect(self.SetPixmap)  # 信号连接槽函数,用来显示图片
        self.AotuRun_thread.signals_result.connect(self.SetResult)  # 信号连接槽函数,用来显示结果
        self.Setup.setEnabled(False)  # 自动运行时屏蔽设置功能
        self.Register.setEnabled(False)
        self.Signout.setEnabled(False)
        self.AutoRun.setEnabled(False)
        self.CloseWindow.setEnabled(False)

    # 登录界面
    def open_login_form(self):
        if self.Login.text() == '登录':
            self.login_form = MyLogin()
            self.login_form.buttonBox.accepted.connect(self.login_fc)
            self.login_form.setWindowModality(QtCore.Qt.ApplicationModal)  # 该模式下，只有该dialog关闭，才可以关闭父界面
            self.login_form.exec_()
        else:
            self.Login.setText('登录')
            global_variable.set_var('login_Admin', False)
            global_variable.set_var('login_Usr', False)
            global_variable.set_var('Usr_name', 0)
            self.Register.setEnabled(False)
            self.Signout.setEnabled(False)
            self.AutoRun.setEnabled(False)

    def login_fc(self):
        # 账号密码便利用户数据库
        account = self.login_form.lineEdit.text()
        password = self.login_form.lineEdit_2.text()
        try:
            with open('data/usrdata.json', 'r') as f:
                data = json.load(f)
            if data['Usr'][account] == password:
                global_variable.set_var('Usr_name', account)
                self.Login.setText('退出')
                QtWidgets.QMessageBox.about(self, "提示", "用户：{} 登录成功！".format(account))
                if account == 'admin':
                    # 管理员登录，权限全开，能注册注销用户
                    global_variable.set_var('login_Admin', True)
                    self.Register.setEnabled(True)
                    self.Signout.setEnabled(True)
                    self.AutoRun.setEnabled(True)
                else:
                    # 用户登录，可以录制人脸
                    global_variable.set_var('login_Usr', True)
                    self.AutoRun.setEnabled(True)
            else:
                QtWidgets.QMessageBox.critical(self, "提示", "密码错误！")
        except:
            QtWidgets.QMessageBox.critical(self, "提示", "登录失败！")

    # 注册界面
    def open_register_form(self):
        self.register_form = MyRegister()
        self.register_form.buttonBox.accepted.connect(self.register_fc)
        self.register_form.setWindowModality(QtCore.Qt.ApplicationModal)  # 该模式下，只有该dialog关闭，才可以关闭父界面
        self.register_form.exec_()

    def register_fc(self):
        # 账号密码便利用户数据库
        account = self.register_form.lineEdit.text()
        password = self.register_form.lineEdit_2.text()
        if len(account) > 0 and len(password) > 0:
            with open('data/usrdata.json', 'r') as f:
                data = json.load(f)
            if account in data['Usr']:
                QtWidgets.QMessageBox.critical(self, "注册失败", "该用户已存在！")
            else:
                data['Usr'][account] = password
                data_json = json.dumps(data)
                with open('data/usrdata.json', 'w') as w:
                    w.write(data_json)
                QtWidgets.QMessageBox.about(self, "注册成功", "用户：{} 已注册！".format(account))
        else:
            QtWidgets.QMessageBox.critical(self, "注册失败", "账号密码不能为空！")

    # 注销界面
    def open_signout_form(self):
        self.signout_form = MySignOut()
        self.signout_form.buttonBox.accepted.connect(self.signout_fc)
        self.signout_form.setWindowModality(QtCore.Qt.ApplicationModal)  # 该模式下，只有该dialog关闭，才可以关闭父界面
        self.signout_form.exec_()

    def signout_fc(self):
        # 账号密码便利用户数据库
        account = self.signout_form.lineEdit.text()
        if len(account) > 0:
            with open('data/usrdata.json', 'r') as f:
                data = json.load(f)
            if account in data['Usr']:
                data['Usr'].pop(account)
                data_json = json.dumps(data)
                with open('data/usrdata.json', 'w') as w:
                    w.write(data_json)
                QtWidgets.QMessageBox.about(self, "注销成功", "用户：{} 已注销！".format(account))

                if os.path.exists(os.path.join("dataset", account)):
                    shutil.rmtree(os.path.join("dataset", account))
            else:
                QtWidgets.QMessageBox.about(self, "注销识别", "用户：{} 不存在！".format(account))
        else:
            QtWidgets.QMessageBox.critical(self, "注销失败", "账号不能为空！")

    # 关闭
    def closeEvent(self, event):
        """Generate 'question' dialog on clicking 'X' button in title bar.
        Reimplement the closeEvent() event handler to include a 'Question'
        dialog with options on how to proceed - Save, Close, Cancel buttons
        """
        reply = QMessageBox.question(
            self, "关闭程序",
            "确定退出程序？.",
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Ok)

        if reply == QMessageBox.Ok:
            app.quit()
        else:
            pass

    # 时间日期
    def clock(self):
        t = time.strftime('%Y-%m-%d %H:%M:%S')
        self.lcdNumber.display(t)

    # 显示图像
    def SetPixmap(self, Pixmap):
        if Pixmap == '':
            # if self.label.isVisible():
            #     self.label.setVisible(False)
            self.label.clear()
        else:
            # if self.label.isVisible() == False:
            #     self.label.setVisible(True)
            self.label.setPixmap(Pixmap)

    # 显示结果
    def SetResult(self, Result):
        if Result == 'model done':
            # 模型训练完毕后恢复按键
            self.result.setText('')
            self.Face_data_get.setText('录入人脸数据')
            self.Manage.setEnabled(True)
            self.Setup.setEnabled(True)
            self.Open_cam.setEnabled(True)
            self.Open_face_regionizer.setEnabled(True)
            self.Train_face_data.setEnabled(True)
        else:
            # 正常显示
            self.result.setText(Result)

    # 自动运行
    def AotuRun_thread_start(self):
        if self.AutoRun.text() == '自动运行':
            self.AutoRun.setText('停止运行')
            self.AotuRun_thread = AutoWorkTread()  # 类的实例化
            self.AotuRun_thread.start()  # 开启线程
            self.AotuRun_thread.signals_img.connect(self.SetPixmap)  # 信号连接槽函数,用来显示图片
            self.AotuRun_thread.signals_result.connect(self.SetResult)  # 信号连接槽函数,用来显示结果
            # 自动运行时屏蔽设置功能和除登录以外的所有案件
            self.Setup.setEnabled(False)
            self.Register.setEnabled(False)
            self.Signout.setEnabled(False)
            self.CloseWindow.setEnabled(False)

        else:
            self.AotuRun_thread.stop()  # 停止线程
            self.AutoRun.setText('自动运行')
            # 停止运行后恢复按键
            self.Setup.setEnabled(True)
            self.Login.setEnabled(True)
            self.Signout.setEnabled(True)
            self.CloseWindow.setEnabled(True)
            self.label.clear()

    # 打开摄像头
    def Open_cam_fc_start_thread(self):
        if self.Open_cam.text() == '开启摄像头':
            self.label.setText('相机启动中....')
            self.Open_cam.setText('关闭摄像头')
            # 开启QT线程
            self.Open_cam_thread = WorkThread1()  # 类的实例化
            self.Open_cam_thread.start()  # 开启线程
            self.Open_cam_thread.signals.connect(self.SetPixmap)  # 信号连接槽函数

            # 启动摄像头后屏蔽其他功能键
            self.Open_face_regionizer.setEnabled(False)
            self.Face_data_get.setEnabled(False)
            self.Train_face_data.setEnabled(False)
        else:
            # 停止线程
            self.Open_cam_thread.stop()
            self.Open_cam.setText('开启摄像头')
            self.Open_cam_thread_isWork = False
            # 关闭摄像头后恢复其他功能键
            self.Open_face_regionizer.setEnabled(True)
            self.Face_data_get.setEnabled(True)
            self.Train_face_data.setEnabled(True)
            self.label.clear()

    # 打开人脸识别
    def Open_face_regionizer_fc_start_thread(self):
        if self.Open_face_regionizer.text() == '启动人脸识别':
            with open('data/usrdata.json', 'r') as f:
                data = json.load(f)
            face_model_fileName = data['Setup']['model_path']
            self.label.setText('人脸模型启动中....')
            self.Open_face_regionizer.setText('关闭人脸识别')
            # 开启QT线程
            self.Open_face_regionizer_thread = WorkThread2(face_model_fileName)
            self.Open_face_regionizer_thread.start()
            self.Open_face_regionizer_thread.signals.connect(self.SetPixmap)
            # 启动后屏蔽其他功能键
            self.Open_cam.setEnabled(False)
            self.Face_data_get.setEnabled(False)
            self.Train_face_data.setEnabled(False)
        else:
            self.Open_face_regionizer_thread.stop()
            self.Open_face_regionizer.setText('启动人脸识别')
            # 关闭后恢复其他功能键
            self.Open_cam.setEnabled(True)
            self.Face_data_get.setEnabled(True)
            self.Train_face_data.setEnabled(True)
            self.label.clear()

    # 录制人脸数据
    def Face_data_get_fc_start_thread(self):
        if self.Face_data_get.text() == '录入人脸数据':
            self.Face_data_get.setText('停止录入数据')
            face_data_name, ok_pressed = QtWidgets.QInputDialog.getText(self, "录入人脸", "请输入人脸名字",
                                                                        QtWidgets.QLineEdit.Normal, "")
            if ok_pressed and face_data_name != '':
                # 开启QT线程
                self.Face_data_get_thread = WorkThread3(face_data_name)
                self.Face_data_get_thread.start()
                self.Face_data_get_thread.signals_img.connect(self.SetPixmap)
                self.Face_data_get_thread.signals_result.connect(self.SetResult)
                # 启动后屏蔽其他功能键
                self.Open_cam.setEnabled(False)
                self.Open_face_regionizer.setEnabled(False)
                self.Train_face_data.setEnabled(False)
        else:
            self.Face_data_get_thread.stop()
            self.Face_data_get.setText('录入人脸数据')
            # 关闭后恢复其他功能键
            self.Open_cam.setEnabled(True)
            self.Open_face_regionizer.setEnabled(True)
            self.Train_face_data.setEnabled(True)
            self.label.clear()

    # 训练人脸模型
    def Train_face_data_fc_start_thread(self):
        # 开启QT线程
        self.Train_face_data_thread = WorkThread4()
        self.Train_face_data_thread.start()
        self.Train_face_data_thread.signals_result.connect(self.SetResult)
        # 屏蔽所有按键
        self.Manage.setEnabled(False)
        self.Setup.setEnabled(False)

    # 错误信息显示
    def show_message(self):
        QtWidgets.QMessageBox.critical(self, "错误", "模型类型错误，请重新选择")


if __name__ == '__main__':
    # QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    MainWindow = MyWindow()
    # 全屏显示
    # MainWindow.showFullScreen()
    MainWindow.show()
    sys.exit(app.exec())
