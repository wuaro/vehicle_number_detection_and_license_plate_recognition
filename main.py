from models.experimental import *
from utils.datasets import *
from utils.utils import *
from models.LPRNet import *
from models.experimental import *
from utils.datasets import *
from utils.utils import *
from models.LPRNet import *
import sys
import cv2
import argparse
import random
import torch
import numpy as np
from PIL import Image
from yolo import YOLO
import torch.backends.cudnn as cudnn
from PyQt5 import QtCore, QtGui, QtWidgets
from models.experimental import attempt_load
from utils.general import non_max_suppression as NON
from utils.general import *
from utils.torch_utils2 import select_device
from models.experimental import *
from utils.datasets import *
from detect import main

#主界面类
class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_logo()
        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.out = None
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='./weights/last.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='./inference/images/',
                            help='source')  # file/folder, 0 for webcam
        parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        self.opt = parser.parse_args()
        print(self.opt)
        self.yolo = YOLO()
        self.out, source, weights, view_img, self.save_txt, imgsz = self.opt.output,self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # 半精度只支持CUDA
        cudnn.benchmark = True

        # 加载模型
        self.model = attempt_load(weights, map_location=self.device)  # 加载 FP32 模型
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())  # 检查图片尺寸
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # 二级分类器
        classify = True
        if classify:
            self.modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(self.device)
            self.modelc.load_state_dict(torch.load('./weights/Final_LPRNet_model.pth', map_location=torch.device('cpu')))
            print("load pretrained model successful!")
            self.modelc.to(self.device).eval()


        # name和color
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    # 主要UI
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1500, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(-1, -1, 0, -1)
        self.verticalLayout.setSpacing(80)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_img = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_img.sizePolicy().hasHeightForWidth())
        self.pushButton_img.setSizePolicy(sizePolicy)
        self.pushButton_img.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_img.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_img.setFont(font)
        self.pushButton_img.setObjectName("pushButton_img")
        self.verticalLayout.addWidget(self.pushButton_img, 0, QtCore.Qt.AlignHCenter)
        self.pushButton_camera = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_camera.sizePolicy().hasHeightForWidth())
        self.pushButton_camera.setSizePolicy(sizePolicy)
        self.pushButton_camera.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_camera.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_camera.setFont(font)
        self.pushButton_camera.setObjectName("pushButton_camera")
        self.verticalLayout.addWidget(self.pushButton_camera, 0, QtCore.Qt.AlignHCenter)
        self.pushButton_video = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_video.sizePolicy().hasHeightForWidth())
        self.pushButton_video.setSizePolicy(sizePolicy)
        self.pushButton_video.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_video.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_video.setFont(font)
        self.pushButton_video.setObjectName("pushButton_video")
        self.verticalLayout.addWidget(self.pushButton_video, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.setStretch(2, 1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    #主界面窗口
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "YOLOv4+LPRNet机动车数量检测及车牌识别"))
        self.pushButton_img.setText(_translate("MainWindow", "图片检测"))
        self.pushButton_camera.setText(_translate("MainWindow", "批量检测"))
        self.pushButton_video.setText(_translate("MainWindow", "返回"))
        self.label.setText(_translate("MainWindow", "TextLabel"))

    def init_slots(self):
        self.pushButton_img.clicked.connect(self.button_image_open)
        self.pushButton_video.clicked.connect(self.button_video_open)
        self.pushButton_camera.clicked.connect(self.button_camera_open)
        #self.timer_video.timeout.connect(self.show_video_frame)

    def init_logo(self):
        pix = QtGui.QPixmap('demo.png')
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)


    #图片检测按钮
    def button_image_open(self):
        print('进入图片检测')
        name_list = []
        # 获取名字和颜色
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
        #选择图片并打开，img_name为图片路径
        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        print("所选图片路径：", img_name)
        #读取选中图片，赋给im0s
        im0s = cv2.imread(img_name)
        #将选中的图片导入数据加载器
        dataset = LoadImages(img_name, img_size=self.imgsz)
        showimg = im0s
        with torch.no_grad():
            img = letterbox(im0s, new_shape=self.opt.img_size)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = self.model(img, augment=self.opt.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            # Apply Classifier
            pred,plat_num = apply_classifier(pred, self.modelc, img, im0s)
            print(pred)
            # 检测代码
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = img_name, '', im0s
                save_path = str(Path(self.out) / Path(p).name)
                txt_path = str(Path(self.out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # 重新缩放框从img_size到im0大小
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    # Write results
                    for de, lic_plat in zip(det, plat_num):
                        *xyxy, conf, cls = de
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, xywh))  # 写标签
                        lb = ""
                        for a, i in enumerate(lic_plat):
                            lb += CHARS[int(i)]
                        label = '%s %.2f' % (lb, conf)
                        #将检测出的车牌号和车牌框画上
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        img_pil = Image.fromarray(im0)  # narray转化为图片
                        im0 = self.yolo.detect_image(img_pil)  # 图片才能检测
                im0 = np.array(im0)  # 图片转化为 narray
                showimg = im0
                #存储最终生成的图片到./inference/output
                cv2.imwrite(save_path, im0)
        # 再生成一份在项目文件夹下，方便显示到界面中
        cv2.imwrite('prediction.jpg', showimg)
        self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
        self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
        self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))

    #返回检测按钮
    def button_video_open(self):
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_logo()
        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.out = None

    # 批量检测按钮
    def button_camera_open(self):

        #显示请等待
        pix = QtGui.QPixmap('please_wait.png')
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)
        m = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", "E:/")  # 起始路径
        #调用detect.main()开始批量检测
        flag = True
        if (flag):
            main(s=m)
            flag = False
        #显示已完成
        if (flag == False):
            pix = QtGui.QPixmap('complete.png')
            self.label.setScaledContents(True)
            self.label.setPixmap(pix)





if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
