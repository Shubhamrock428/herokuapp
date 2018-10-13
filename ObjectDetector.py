import cv2 as cv
import numpy


classNames = {1: 'match2', 2: 'match3', 3: 'match4', 4: 'match5', 5: 'match6', 6: 'match7',
              7: 'match8', 8: 'match9', 9: 'match10', 10: 'match12', 11: 'match13',
              12: 'match14', 13: 'match15', 14: 'match16', 15: 'match17', 16: 'match18',
              17: 'match19', 18: 'match20', 19: 'match21', 20: 'match23', 21: 'match24', 22: 'match25',
              23: 'match26', 24: 'match27', 25: 'match28', 26: 'match29', 27: 'match30',
              28: 'match31', 29: 'match32', 30: 'match33', 31: 'match34', 32: 'match35',
              33: 'match36', 34: 'match37', 35: 'match38', 36: 'match39',
              37: 'match40', 38: 'match42', 39: 'match43', 40: 'match44',
              41: 'match45', 42: 'match46', 43: 'match47', 44: 'match48', 45: 'match49',
              46: 'match51', 47: 'match53', 48: 'match54', 49: 'match55', 50: 'match56',
              51: 'match57', 52: 'match58', 53: 'match59', 54: 'match60', 55: 'match61',
              56: 'match62', 57: 'match63', 58: 'match64', 59: 'match65', 60: 'match66',
              61: 'match67', 62: 'match68', 63: 'match69', 64: 'match70', 65: 'match71',
              66: 'match72', 67: 'match73', 68: 'match74', 69: 'match75', 70: 'match76',
              71: 'match77', 72: 'match78', 73: 'match79', 74: 'match80', 75: 'match81',
              76: 'match82', 77: 'match83', 78: 'match84', 79: 'match85',80: 'match86',
              81: 'match87', 82: 'match88', 83: 'match89', 84: 'match90', 85: 'match91',
              86: 'match92', 87:'match93', 88: 'match94', 89:'match95'}


class Detector:
    def __init__(self):
        global cvNet
        cvNet = cv.dnn.readNetFromTensorflow('/model/frozen_inference_graph.pb',
                                             '/model/graph.pbtxt')

    def detectObject(self, imName):
        img = cv.cvtColor(numpy.array(imName), cv.COLOR_BGR2RGB)
        #img = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)
        cvNet.setInput(cv.dnn.blobFromImage(img, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        detections = cvNet.forward()
        cols = img.shape[1]
        rows = img.shape[0]

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])

                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)

                cv.rectangle(img, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                             (0, 0, 255))
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv.putText(img, label, (xLeftBottom+5, yLeftBottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
                    
          img = cv.imencode('.jpg', img)[1].tobytes()
          return img
