import cv2

def img_resize(img, scale):
    img = cv2.resize(
        src = img,
        dsize = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    )

    return img

def img_toGrayscale(img):
    img = cv2.cvtColor(
        src = img, 
        code = cv2.COLOR_BGR2GRAY
    )

    return img

def img_crop(img, pt1, pt2, out_dsize = (200, 200), opt = "SQUARED_PADDING"):
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    
    if opt == "SQUARED_PADDING":
        delta = abs(w - h)

        if w > h:
            y1 -= delta // 2
            y2 += delta // 2 + delta % 2
        else:
            x1 -= delta // 2
            x2 += delta // 2 + delta % 2

    if x1 < 0 or y1 < 0:
        return None
    
    return cv2.resize(
        src = img[y1:y2, x1:x2],
        dsize = out_dsize
        )