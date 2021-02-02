import sys
from scipy.spatial import distance as dist
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi
import cv2
from imutils import perspective
from imutils import contours
import imutils
import numpy as np

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

class ColorDetector(QDialog):
    def __init__(self):
        super(ColorDetector, self).__init__()
        loadUi('OpenCV.ui', self)
        self.image = None
        self.start_button.clicked.connect(self.start_webcam)
        self.stop_button.clicked.connect(self.stop_webcam)
        
        self.track_button.setCheckable(True)
        self.track_button.toggled.connect(self.track_webcam)
        self.track_enabled = False
        
        
    def track_webcam(self, status):
        if status:
            self.track_enabled = True
            self.track_button.setText('Stop Tracking')
        else:
            self.track_enabled = False
            self.track_button.setText('Track Color')    
    
    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
       
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)
        
    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image,1)
        self.displayImage(self.image,1)
        
        #Reference
        #lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'blue':(97, 100, 117), 'yellow':(23, 59, 119), 'orange':(0, 50, 80)} #assign new item lower['blue'] = (93, 10, 0)
        #upper = {'red':(186,255,255), 'green':(86,255,255), 'blue':(117,255,255), 'yellow':(54,255,255), 'orange':(20,255,255)}
        
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0,120,70])
        upper_red1 = np.array([5,255,255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([170,120,70])
        upper_red2 = np.array([180,255,255])

        mask2 = cv2.inRange(hsv,lower_red2,upper_red2)

    
        mask = mask1+mask2
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel)

        self.displayImage(mask,2)
        
        if(self.track_enabled):
            trackedImage = self.track_colored_object(self.image.copy())
            self.displayImage(trackedImage,1)
        else:
            self.displayImage(self.image,1)
        
    def track_colored_object(self,img):
        

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    
        lower_red1 = np.array([0,120,70])
        upper_red1 = np.array([5,255,255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([170,120,70])
        upper_red2 = np.array([180,255,255])

        mask2 = cv2.inRange(hsv,lower_red2,upper_red2)

    
        mask = mask1+mask2
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel)

    
        if int(cv2.__version__[0]) > 3:
        
            cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
        
             _, cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        pixelsPerMetric = None

    
        for c in cnts:
        
            area = cv2.contourArea(c)
            if area > 5000:
                approx = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
                x = approx.ravel()[0]
                y = approx.ravel()[1]
                
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")

        
                box = perspective.order_points(box)
                cv2.drawContours(img, [box.astype("int")], -1, (0, 255, 0), 2)
     
        
                for (x, y) in box:
                    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

    
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
     
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)
     
        
                cv2.circle(img, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                cv2.circle(img, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                cv2.circle(img, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                cv2.circle(img, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
     
    
                cv2.line(img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                    (255, 0, 255), 2)
                cv2.line(img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                    (255, 0, 255), 2)

                if len(approx) == 4:
                    cv2.putText(img, "Persegi", (x, y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
                    #self.Bentuk.setChecked(True)
                #else:
                    #elf.Bentuk.setChecked(False)

                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))


     

        
                dimA = (dA  * 0.026458) 
                dimB = (dB * 0.026458) 
                dimC = (dimA * dimB)


        
                cv2.putText(img, "{:.1f}cm".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
                cv2.putText(img, "{:.1f}cm".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

                if (len(approx) == 4 ) and (area > 5000) and (dimC > 9 ) and (dimC < 12):
                    self.Bentuk.setChecked(True)
                    self.Warna.setChecked(True)
                    self.Luas.setChecked(True)
                else :
                    self.Bentuk.setChecked(False)
                    self.Warna.setChecked(False)
                    self.Luas.setChecked(False)

        return img
        
    def stop_webcam(self):
        self.capture.release()
        self.timer.stop()
        
    
    def displayImage(self,img,window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3: #[0]=rows, [1]=cols, [2]=channels
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0],qformat)
        #BGR to RGB
        outImage = outImage.rgbSwapped()
        
        if window == 1:
            self.image_label1.setPixmap(QPixmap.fromImage(outImage))
            self.image_label1.setScaledContents(True)
        
        if window ==2:
            self.image_label2.setPixmap(QPixmap.fromImage(outImage))
            self.image_label2.setScaledContents(True)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ColorDetector()
    window.setWindowTitle('OpenCV Color Detector')
    window.show()
    sys.exit(app.exec_())