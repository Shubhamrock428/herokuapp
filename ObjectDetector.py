import cv2 
import numpy


classNames = {3.1: 'match2', 3.2: 'match3', 3.3: 'match4', 3.4: 'match5', 3.5: 'match6',
              3.6: 'match7', 3.7: 'match8', 3.8: 'match9', 3.9: 'match10', 3.10: 'match12',
              3.11: 'match13', 3.12: 'match14', 3.13: 'match15', 3.14: 'match16', 3.15: 'match17',
              3.16: 'match18', 3.17: 'match19', 3.18: 'match20', 3.19: 'match21', 3.20: 'match23', 
              3.21: 'match24', 3.22: 'match25', 3.23: 'match26', 3.24: 'match27', 3.25: 'match28',
              3.26: 'match29', 3.27: 'match30', 3.28: 'match31', 3.29: 'match32', 3.30: 'match33',
              3.31: 'match34', 3.32: 'match35', 3.33: 'match36', 3.34: 'match37', 3.35: 'match38', 
              3.36: 'match39', 3.37: 'match40', 3.38: 'match42', 3.39: 'match43', 3.40: 'match44',
              3.41: 'match45', 3.42: 'match46', 3.43: 'match47', 3.44: 'match48', 3.45: 'match49',
              3.46: 'match51', 3.47: 'match53', 3.48: 'match54', 3.49: 'match55', 3.50: 'match56',
              3.51: 'match57', 3.52: 'match58', 3.53: 'match59', 3.54: 'match60', 3.55: 'match61',
              3.56: 'match62', 3.57: 'match63', 3.58: 'match64', 3.59: 'match65', 3.60: 'match66',
              3.61: 'match67', 3.62: 'match68', 3.63: 'match69', 3.64: 'match70', 3.65: 'match71',
              3.66: 'match72', 3.67: 'match73', 3.68: 'match74', 3.69: 'match75', 3.70: 'match76',
              3.71: 'match77', 3.72: 'match78', 3.73: 'match79', 3.74: 'match80', 3.75: 'match81',
              3.76: 'match82', 3.77: 'match83', 3.78: 'match84', 3.79: 'match85', 3.80: 'match86',
              3.81: 'match87', 3.82: 'match88', 3.83: 'match89', 3.84: 'match90', 3.85: 'match91',
              3.86: 'match92', 3.87: 'match93', 3.88: 'match94', 3.89: 'match95'}


class Detector:
    def __init__(self):
        global cv2Net
        cv2Net = cv2.dnn.readNetFromTensorflow('model/frozen_inference_graph.pb',
                                               'model/labelmap.pbtxt')

    def detectObject(self, imName):
        img = cv2.cvtColor(numpy.array(imName), cv2.COLOR_BGR2RGB)
        cv2Net.setInput(cv2.dnn.blobFromImage(img, 0.007843, (300,300), (127.5, 127.5, 127.5),swapRB=True, crop=False))
        detections = cv2Net.forward()
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

                cv2.rectangle(img, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                             (0, 0, 255))
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.putText(img, label, (xLeftBottom+5, yLeftBottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

        img = cv2.imencode('.jpg', img)[1].tobytes()
        return img
