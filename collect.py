import sys

import cv2
import numpy as np

from utils.preprocess import *
        
DATA_PATH = "./data/"
RAW_IMG_W = 640
RAW_IMG_H = 480

def main(args):
    # SET UP MODEL

    model = cv2.FaceDetectorYN.create("face_detection_yunet_2022mar.onnx", "", (320, 320))
    model.setInputSize([RAW_IMG_W, RAW_IMG_H])

    webcam = cv2.VideoCapture(0)

    student_ID = args[0]

    window_name = "Face capturing for {}".format(student_ID)

    img_counter = int(args[1])
    flg = False
    while img_counter < int(args[1]) + int(args[2]):
        ret, frame = webcam.read()
        if not ret:
            print("Something is wrong!")
            break

        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1)
        if key % 256 == 32:
            if flg:
                print("Stop capturing!")
                flg = False
            else:
                print("Start capturing ...")
                flg = True
        elif key % 256 == 27:
            print("Stop collecting by user ...")
            break
        
        if flg:
            frame = img_resize(frame, 1)
            num_faces, face_details = model.detect(frame)
            if face_details is not None:
                face_details = face_details.astype("int64")
                frame = img_toGrayscale(frame)
                face_x1, face_y1 = face_details[0][0], face_details[0][1]
                face_x2, face_y2 = face_details[0][0] + face_details[0][2], face_details[0][1] + face_details[0][3]
                if face_x1 < 0:
                    face_x1 = 0
                if face_y1 < 0:
                    face_y1 = 0
                if face_x2 >= RAW_IMG_W:
                    face_x2 = RAW_IMG_W - 1
                if face_y2 >= RAW_IMG_H:
                    face_y2 = RAW_IMG_H - 1
                
                frame = img_crop(frame, (face_x1, face_y1), (face_x2, face_y2))

                if frame is None:
                    continue
                
                img_name = DATA_PATH + "{}_{}.png".format(student_ID, img_counter)
                cv2.imwrite(img_name, frame)

                print("Captured img {}!".format(img_counter))
                img_counter += 1

    print("Stopped collecting!")
    webcam.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1:])
