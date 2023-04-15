import cv2 as cv
import numpy as np

def img_resize(img, scale):
    img = cv.resize(
        src = img,
        dsize = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    )

    return img

def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for det in (results[1] if results is not None else []):
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        conf = det[-1]
        cv.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)

    return output


image = img_resize(cv.imread("./rawimage/corgi.png"), 0.25)
h, w, _ = image.shape

model = cv.FaceDetectorYN.create("face_detection_yunet_2022mar.onnx", "", (320, 320))
# Inference
model.setInputSize([w, h])
results = model.detect(image)

# Print results
print('{} faces detected.'.format(results[0]))
for idx, det in enumerate(results[1]):
    print('{}: {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}'.format(
        idx, *det[:-1])
    )

# Draw results on the input image
image = visualize(image, results)
cv.imshow("", image)
cv.waitKey(0)
