import torch
from model.model import FaceRecognizer

import cv2
import pandas as pd

import sys
from utils.preprocess import *

MODEL_STATE_DICT_PATH = "./model/model.pt"
MODEL_CLASS2ID = "./model/class2id.csv"
RAW_IMG_W = 640
RAW_IMG_H = 480

def convert_class2id(path):
    class2id = {}
    tmp = pd.read_csv(path)

    i = 0
    for column in tmp.columns:
        class2id[tmp.iloc[0, i]] = column
        i += 1
    
    return class2id

def main(argv):
    # Translate Classes from model to IDs
    class2id = convert_class2id(MODEL_CLASS2ID)

    # Load model state dict
    num_classes = len(class2id.keys())
    model = FaceRecognizer(num_classes = num_classes)
    model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH))
    model.eval()

    # Load pre-trained face detection model
    face_detector = cv2.FaceDetectorYN.create("face_detection_yunet_2022mar.onnx", "", (320, 320))
    face_detector.setInputSize([RAW_IMG_W, RAW_IMG_H])

    # Init camera
    webcam = cv2.VideoCapture(0)

    if argv[0] == "input":
        test_image = cv2.imread(argv[1])
        test_image = img_resize(test_image, 1)

        face_detector.setInputSize([test_image.shape[1], test_image.shape[0]])
        raw_test_image = test_image.copy()
        _, face_details = face_detector.detect(test_image)

        if face_details is not None:
            face_details = face_details.astype("int64")
            face_x1, face_y1 = face_details[0][0], face_details[0][1]
            face_x2, face_y2 = face_details[0][0] + face_details[0][2], face_details[0][1] + face_details[0][3]

            test_image = img_toGrayscale(test_image)
            test_image = img_crop(test_image, (face_x1, face_y1), (face_x2, face_y2))
            test_image = torch.from_numpy(test_image).reshape([-1, 1, 200, 200]).to(torch.float64) / 255

            with torch.no_grad():
                result = model(test_image)
                softmax_layer = torch.nn.Softmax(1)
                result = softmax_layer(result)

                _, index = torch.max(result, axis = 1)

                label_predict = class2id[index.item()]
                prob = "{prob:.2f}%".format(prob = result[0, index].item() * 100)

                raw_test_image = cv2.rectangle(raw_test_image, (face_x1, face_y1), (face_x2, face_y2), color = (255, 0, 0), thickness = 3)
                raw_test_image = cv2.putText(raw_test_image, label_predict, (face_x1, face_y1 - 10), 0, 0.75, (0, 255, 0), thickness = 2)
                raw_test_image = cv2.putText(raw_test_image, prob, (face_x1 + 100, face_y1 - 10), 0, 0.75, (0, 255, 0), thickness = 2)
                cv2.imshow("This is {}".format(label_predict), raw_test_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("Cannot detect face!")

    elif argv[0] == "webcam":
        while True:
            # Escape loop when press "Space"
            key_pressed = cv2.waitKey(1)

            if key_pressed % 256 == 32:
                break
            
            # Read image from webcam
            _, frame = webcam.read()
            test_image = frame
            raw_test_image = test_image.copy()

            # Detect face 
            num_faces, face_details = face_detector.detect(test_image)

            if face_details is not None:
                face_details = face_details.astype("int64")
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

                test_image = img_toGrayscale(test_image)
                test_image = img_crop(test_image, (face_x1, face_y1), (face_x2, face_y2))

                if test_image is None:
                    cv2.imshow("Face detector/Test", raw_test_image)
                    continue

                test_image = torch.from_numpy(test_image).reshape([-1, 1, 200, 200]).to(torch.float64) / 255

                with torch.no_grad():
                    result = model(test_image)
                    softmax_layer = torch.nn.Softmax(1)
                    result = softmax_layer(result)

                    _, index = torch.max(result, axis = 1)

                    label_predict = class2id[index.item()]
                    prob = "{prob:.2f}%".format(prob = result[0, index].item() * 100)

                    raw_test_image = cv2.rectangle(raw_test_image, (face_x1, face_y1), (face_x2, face_y2), color = (255, 0, 0), thickness = 3)
                    raw_test_image = cv2.putText(raw_test_image, label_predict, (face_x1, face_y1 - 10), 0, 0.75, (0, 255, 0), thickness = 2)
                    raw_test_image = cv2.putText(raw_test_image, prob, (face_x1 + 100, face_y1 - 10), 0, 0.75, (0, 255, 0), thickness = 2)
                    cv2.imshow("Face detector/Test", raw_test_image)
            else:
                cv2.imshow("Face detector/Test", raw_test_image)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1:])