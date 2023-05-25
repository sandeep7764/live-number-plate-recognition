__author__ = 'Sandeepa H A'
import numpy as np
import cv2
import imutils
import pytesseract
import pandas as pd
import time

def process_frame(frame):
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 170, 200)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCnt = None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            break

    if NumberPlateCnt is not None:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)

        config = ('-l eng --oem 1 --psm 3')
        text = pytesseract.image_to_string(new_image, config=config)

        cv2.imshow('Number Plate', new_image)

        return text

    return ""

cap = cv2.VideoCapture(0)

data = {'date': [], 'v_number': []}

try:
    df = pd.read_csv('data.csv')  # Load existing data
    data['date'] = df['date'].tolist()
    data['v_number'] = df['v_number'].tolist()
except FileNotFoundError:
    pass

while True:
    ret, frame = cap.read()

    if not ret:
        break

    text = process_frame(frame)

    if text:
        print("Number Plate:", text)
        data['date'].append(time.asctime(time.localtime(time.time())))
        data['v_number'].append(text)

        df = pd.DataFrame(data)
        df.to_csv('data.csv', index=False)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
