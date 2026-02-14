import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
import math
from imutils import rotate_bound
from PIL import Image
from rapidfuzz import process, fuzz


def sort_boxes_reading_order(boxes, line_threshold=100):
    # Sort top-to-bottom
    boxes = sorted(zip(boxes.xyxy, boxes.cls), key=lambda b: (b[0][1] + b[0][3]) // 2)
    heights = list(map(lambda b: b[0][3] - b[0][1], boxes))
    if heights:
        line_threshold = int(sum(heights) / len(heights)) //2

    lines = []
    current_line = []
    current_y = None

    for box, cls in boxes:
        cy = (box[1] + box[3]) // 2  # center Y

        if current_y is None or abs(cy - current_y) < line_threshold:
            current_line.append((box, cls))
            current_y = (current_y + cy) / 2 if current_y else cy
        else:
            # Sort the previous line left-to-right
            lines.append(sorted(current_line, key=lambda b: (b[0][0] + b[0][2]) // 2))
            current_line = [(box, cls)]
            current_y = cy

    # Add last line
    if current_line:
        lines.append(sorted(current_line, key=lambda b: (b[0][0] + b[0][2]) // 2))

    # Flatten the result
    # return [box for line in lines for box in line]
    return lines

#pad and resize
def padded_resize(image: np.ndarray, fill_color=(255, 255, 255)) -> np.ndarray:
    h, w = image.shape[:2]
    # print(h,w)
    imgsz = 800

    if w == h:
        square = image
    elif w > h:
        # Pad height
        pad_vert = w - h
        top = pad_vert // 2
        bottom = pad_vert - top
        square = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=fill_color)
    else:
        # Pad width
        pad_horz = h - w
        left = pad_horz // 2
        right = pad_horz - left
        square = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=fill_color)

    if max(square.shape[:2]) <= 640:
        # Pad up to 640 if smaller
        pad_size = imgsz - square.shape[0]
        top = pad_size // 2
        bottom = pad_size - top

        pad_size = imgsz - square.shape[1]
        left = pad_size // 2
        right = pad_size - left

        final = cv2.copyMakeBorder(square, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)
    else:
        # Resize down to 640x640
        final = cv2.resize(square, (640, 640), interpolation=cv2.INTER_AREA)

    return final

def rotate_using_micr(image, cheque_detect_model):
    image = cv2.resize(image, (2688, 1512))
    result = cheque_detect_model(image, conf=0.7)
    global cheque
    cheque = None

    for i, cls in enumerate(result[0].boxes.cls):
      name = cheque_detect_model.names[int(cls)]
      if name == 'cheque':
        cheqbox = result[0].boxes.xyxy[i]
        x1, y1, x2, y2 = map(int, cheqbox)
        cheq = image[y1:y2, x1:x2]

        r = 7.5/3.5
        w = x2-x1
        h = y2-y1

        angle = 90-math.degrees(math.atan((r*w-h)/(r*h-w)))
        center = w //2, h//2
        output = None
        output1 = rotate_bound(image, angle)
        result1 = cheque_detect_model(output1)
        #result1[0].show()
        ratio1 = None
        ratio2 = None
        for j, cls in enumerate(result1[0].boxes.cls):
          name1 = cheque_detect_model.names[int(cls)]
          if name1 == 'cheque':

            x1, y1, x2, y2 = map(int, result1[0].boxes.xyxy[i])
            ratio1 = (x2-x1)/(y2-y1)



        output2 = rotate_bound(image, -angle)
        result2 = cheque_detect_model(output2)
        #result2[0].show()

        for j, cls in enumerate(result2[0].boxes.cls):
          name1 = cheque_detect_model.names[int(cls)]
          if name1 == 'cheque':

            x1, y1, x2, y2 = map(int, result2[0].boxes.xyxy[i])
            ratio2 = (x2-x1)/(y2-y1)


        result = result1 if ratio1 > ratio2 else result2
        output = output1 if ratio1 > ratio2 else output2



    for i, cls in enumerate(result[0].boxes.cls):
      name = cheque_detect_model.names[int(cls)]
      if name == 'cheque':
        chequebox = result[0].boxes.xyxy[i]
        x1, y1, x2, y2 = map(int, chequebox)
        cheque = output



    cx1, cy1, cx2, cy2 = map(int, chequebox)
    w = cx2-cx1
    h = cy2-cy1

    ax1 = int(cx1 + 0.6*w)
    ay1 = int(cy1 + 0.2*h)
    ay2 = int(cy1 + 0.5*h)
    amtbb = None
    amtbb = cheque[ay1:ay2, ax1:cx2]

    amtgray = cv2.cvtColor(amtbb, cv2.COLOR_BGR2GRAY)

    # plt.imshow(amtgray, cmap='gray')
    # plt.show()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    amtc = clahe.apply(amtgray)

    amtblur = cv2.GaussianBlur(amtc, (5, 5), 0)

    # plt.imshow(amtblur, cmap='gray')
    # plt.show()

    amth = cv2.adaptiveThreshold(
            amtblur,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=31,
            C=10
            )

    # plt.imshow(amth, cmap='gray')
    # plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.morphologyEx(amth, cv2.MORPH_CLOSE, kernel, iterations=1)
    # plt.imshow(dilated, cmap='gray')
    # plt.show()

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ext_idx = max(
        range(len(contours)),
        key=lambda i: cv2.contourArea(contours[i])
        if hierarchy[0][i][3] == -1 else 0
    )

    internal = [
        contours[i]
        for i in range(len(contours))
        if hierarchy[0][i][3] == ext_idx
    ]

    cnt = max(internal, key=cv2.contourArea)

    temp = amtbb.copy()

    mask = np.zeros_like(temp)

    cv2.drawContours(temp, [cnt], -1, (0,255,0), 2,)
    # plt.imshow(temp)
    # plt.show()

    cv2.drawContours(mask, [cnt], contourIdx=0, color=(255,255,255), thickness=-1)

    maskgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    masker = cv2.erode(maskgray, kernel, iterations=2)
    # plt.imshow(masker, cmap='gray')
    # plt.show()

    mcontours, _ = cv2.findContours(masker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # pick largest connected region
    mcnt = max(mcontours, key=cv2.contourArea)
    rect = cv2.minAreaRect(mcnt)

    cv2.drawContours(temp, [cnt], -1, (0,255,0), 2,)

    # plt.imshow(masker, cmap='gray')
    # plt.show()
    cv2.drawContours(temp, [mcnt], -1, (0,0,255), 2)
    #plt.imshow(temp)
    #plt.show()


    # plt.imshow(dilated, cmap='gray')
    # plt.show()

    (xc, yc), (width, height), angle = rect

    print(angle)
    print('width', width, 'height', height)

    if width < height:
      angle = -(90 - angle)
      if angle < -150:
          angle += 180
      width, height = height, width
      print('wield')

    print('angles muji',angle)

    (h, w) = cheque.shape[:2]
    center = w //2, h//2
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    output = cv2.warpAffine(cheque, m, (w, h))

    xc, yc = xc+ax1, yc+ay1

    # print('old', xc, yc)

    xc, yc = map(int, m @ np.array([xc, yc, 1]))
    # print('new', xc , yc)


    cheque = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    #plt.imshow(cheque)
    #plt.show()

    return cheque, (xc, yc), (width, height)

def extract_roi(cheque, amtcenter, amtsize):
    xc, yc = amtcenter
    width, height = amtsize
    sfx = width / 2
    sfy = height / (3/8)
    sf = 0.5 * (sfx + sfy)
    print('factor', sfx, sfy)

    # #amount box
    amountx1 = int(xc - width/2)
    amountx2 = int(xc + width/2)
    amounty2 = int(yc + height/2)
    amounty1 = int(yc - height/2)
    # print(amountx1, amountx2, amounty1, amounty2)

    amt = cheque[amounty1+10:amounty2-10, amountx1+10:amountx2-10]
    #plt.imshow(amt)
    #plt.show()
    gray = cv2.cvtColor(amt, cv2.COLOR_BGR2GRAY)
    amt_gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # plt.imshow(micr_gray, cmap='gray')
    # plt.show()

    amt_bin = cv2.adaptiveThreshold(
            amt_gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=31,
            C=10
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    dilated = cv2.dilate(amt_bin, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # pick largest connected region
    cnt = max(contours, key=cv2.contourArea)

    tmp = amt.copy()
    mx,my,mw,mh = cv2.boundingRect(cnt)
    amt = tmp[my:my+mh, mx+25:mx+mw-25]

    # #signature boxes
    signbx1 = amountx1 - int(sf * (1+7/8))
    signbx2 = amountx1
    signby1 = amounty1 + int(sf * (0.5))
    signby2 = amounty1 + int(sf * (1.5))

    signatureb = cheque[signby1:signby2, signbx1:signbx2]
    # plt.imshow(signatureb)
    # plt.show()

    signax1 = amountx1 + int(sf * (1/8))
    signax2 = amountx1 + int(sf * (2))
    signay1 = amounty1 + int(sf * (0.5))
    signay2 = amounty1 + int(sf * (1.5))

    signaturea = cheque[signay1:signay2, signax1:signax2]
    # plt.imshow(signaturea)
    # plt.show()

    # #date
    datex1 = amountx1
    datex2 = amountx2
    datey1 = amounty1 - int(sf * (7/8))
    datey2 = amounty1 - int(sf * (0.5))

    date = cheque[datey1:datey2, datex1:datex2]
    # #name
    namex1 = amountx1 - int(sf * (3+7/8))
    namex2 = amountx1 + int(sf * (1.5))
    namey1 = amounty1 - int(sf * (0.5))
    namey2 = amounty1

    name = cheque[namey1:namey2, namex1:namex2]
    # plt.imshow(name)
    # plt.show()

    # #amount in words
    wordsx1 = amountx1 - int(sf * (5))
    wordsx2 = amountx1
    wordsy1 = amounty1 - int(sf * (1/8))
    wordsy2 = amounty1 + int(sf * (3/8))

    words = cheque.copy()[wordsy1:wordsy2, wordsx1:wordsx2]
    #neglecting printed texts
    words[:int(sf * (2.5/8)), :int(sf * (3/8))] = 0
    words[int(sf* (2/8)):int(sf* (3.5/8)), int(sf * (3+6/8)):int(sf * (4.5))] = 0
    #plt.imshow(words)
    #plt.show()

    # #micr cheque
    micrx1 = amountx1 - int(sf * (5+5/8))
    micrx2 = amountx1
    micry1 = amounty1 + int(sf * (1+5/8))
    micry2 = amounty1 + int(sf * (2+2/8))

    micr = cheque[micry1:micry2, micrx1:micrx2]

    micr_gray = cv2.cvtColor(micr, cv2.COLOR_BGR2GRAY)
    micr_gray = cv2.GaussianBlur(micr_gray, (7, 7), 0)

    # plt.imshow(micr_gray, cmap='gray')
    # plt.show()

    micr_bin = cv2.adaptiveThreshold(
            micr_gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=31,
            C=10
        )

    # plt.imshow(micr_bin, cmap='gray')
    # plt.show()
    # plt.imshow(micr)
    # plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    dilated = cv2.dilate(micr_bin, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # pick largest connected region
    cnt = max(contours, key=cv2.contourArea)

    tmp = micr.copy()
    mx,my,mw,mh = cv2.boundingRect(cnt)
    # plt.imshow(tmp[my-10:my+mh+10, mx:mx+mw])
    # plt.show()

    micr = tmp[my-15:my+mh+15, mx:mx+mw]


    return [amt,  name, words, date, micr]

def parse_micr(micr):
    nums = re.findall(r'\d+', micr)
    return nums

def extract_date(date_img, ocr_model, language=0):
  img = cv2.cvtColor(date_img, cv2.COLOR_BGR2GRAY)
  img = cv2.adaptiveThreshold(
            img,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=31,
            C=10
        )
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

  img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
  h, w = img.shape
  text = ''
  for i in range(8):
    box = date_img[10:h-10, int(i*w/8+10):int((i+1)*w/8-10)]
    #plt.imshow(box, cmap='gray')
    #plt.show()
    # cls=1
    pred_text = ocr_model.predict(box, num_bias=3)
    if language==0:
        if len(pred_text)>1 and pred_text != '11':
            pred_text = ''.join(pred_text.split('1'))

    text += pred_text
    print(pred_text)
  return text[:2]+'-'+text[2:4]+'-'+text[4:] if len(text)>=8 else None

def code2name(bank_code):
    codes = bank_codes = {
        '0001': 'Nepal Rastra Bank',

        '0101': 'Nepal Bank Limited',
        '0201': 'Rastriya Banijya Bank Limited',

        '0223': 'City Express Money Transfer Pvt. Ltd.',
        '0224': 'GME Remit Pvt.Ltd.',
        '0225': 'Citizen Life Insurance Company Limited',
        '0226': 'Union Life Insurance Company Limited',
        '0227': 'IME Life Insurance Company Ltd.',

        '0301': 'Agriculture Development Bank Ltd',
        '0333': 'Global IME Capital Ltd.',

        '0401': 'Nabil Bank Ltd.',
        '0501': 'Nepal Investment Bank Limited',
        '0601': 'Standard Chartered Bank Nepal Limited',
        '0701': 'Himalayan Bank Limited',
        '0801': 'Nepal SBI Bank Limited',
        '0901': 'Nepal Bangladesh Bank Ltd',
        '1001': 'Everest Bank Limited',
        '1101': 'Bank of Kathmandu Limited',
        '1201': 'Nepal Credit and Commerce Bank Limited',

        '1501': 'Machhapuchre Bank Limited',
        '1601': 'Kumari Bank Ltd.',
        '1701': 'Laxmi Bank Ltd',
        '1801': 'Siddhartha Bank Limited',
        '1901': 'Global IME Bank Limited',
        '2001': 'Citizens Bank International Limited',
        '2101': 'Prime Commercial Bank Limited',
        '2201': 'Sunrise Bank Limited',
        '2301': 'NIC Asia Bank Limited',

        '2501': 'NMB Bank Limited',
        '2601': 'Prabhu Bank Limited',
        '2801': 'Mega Bank Ltd',
        '3001': 'Civil Bank Ltd.',
        '3101': 'Century Commercial Bank Limited',
        '4501': 'Sanima Bank Ltd.',

        '5401': 'Lumbini Bikas Bank Ltd',
        '5901': 'Saptakoshi Development Bank Ltd.',
        '6001': 'Jyoti Bikash Bank Ltd',
        '6401': 'Sindhu Bikash Bank Ltd.',
        '6601': 'Garima Bikas Bank Limited',
        '6801': 'Kamana Sewa Bikas Bank Ltd',
        '6901': 'Gandaki Bikas Bank Ltd',
        '7201': 'Muktinath Bikas Bank Limited',
        '7301': 'Shangrila Development Bank Limited',

        '7502': 'Excel Development Bank Ltd',
        '7509': 'Miteri Development Bank Limited',
        '7516': 'Best Finance Company Ltd',
        '7517': 'Green Development Bank Ltd',
        '7518': 'Nepal Infrastructure Bank Ltd.',

        '7604': 'Janaki Finance Company Ltd.',
        '7606': 'Sahara Bikash Bank Limited',

        '8101': 'Shine Resunga Development Bank Ltd.',
        '8201': 'Srijana Finance Limited',
        '8301': 'Gurkhas Finance Limited',
        '8501': 'Union Finance Ltd.',
        '9001': 'Goodwill Finance Limited',
        '9201': 'Shree Investment & Finance Co. Ltd.',
        '9701': 'Lalitpur Finance Co. Ltd',
        '9801': 'United Finance Ltd.',

        '9902': 'Progressive Finance Co. Ltd.',
        '9905': 'Pokhara Finance Ltd.',
        '9906': 'Central Finance Ltd',
        '9908': 'Corporate Development Bank Ltd',
        '9911': 'Samriddhi Finance Company Ltd.',
        '9915': 'Guheshwori Merchant Banking and Finance',
        '9919': 'ICFC Finance Limited',
        '9929': 'Karnali Bikas Bank Ltd',
        '9931': 'Mahalaxmi Bikash Bank Ltd.',
        '9935': 'Manjushree Finance Limited',
        '9939': 'Reliance Finance Ltd.',
    }
    if len(bank_code)>4:
        bank_code = bank_code[:4]
    return codes.get(bank_code, f'Unknown bank code: {bank_code}')


def normalize_words_eng(text):
    VALID_WORDS = [
        "zero","one","two","three","four","five","six","seven","eight","nine",
        "ten","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety",
        "hundred","thousand","lakh","crore","only"
    ]

    tokens = text.lower().split()
    out = []

    for t in tokens:
        match, score, _ = process.extractOne(
            t, VALID_WORDS, scorer=fuzz.ratio
        )
        out.append(match if score > 70 else '')

    return " ".join(out)


def normalize_words_nep(text):
    VALID_WORDS = ['एक', 'दुई', 'तीन', 'चार', 'पाँच', 'छ', 'सात', 'आठ', 'नौ', 'दश', 'एघार', 'बाह्र', 'तेह्र', 'चौध', 'पन्ध्र', 'सोह्र', 'सत्र', 'अठार', 'उन्नाइस', 'बीस', 'एक्काइस', 'बाइस', 'तेइस', 'चौबीस', 'पच्चीस', 'छब्बीस', 'सत्ताइस', 'अठ्ठाइस', 'उनन्तीस', 'तीस', 'एकतीस', 'बत्तीस', 'तेत्तीस', 'चौँतीस', 'पैंतीस', 'छत्तीस', 'सैंतीस', 'अठतीस', 'उनन्चालीस', 'चालीस', 'एकचालीस', 'बयालीस', 'त्रियालीस', 'चवालीस', 'पैंतालीस', 'छयालीस', 'सच्चालीस', 'अठचालीस', 'उनन्चास', 'पचास', 'एकाउन्न', 'बाउन्न', 'त्रिपन्न', 'चउन्न', 'पचपन्न', 'छपन्न', 'सन्ताउन्न', 'अन्ठाउन्न', 'उनन्साठी', 'साठी', 'एकसट्ठी', 'बयसट्ठी', 'त्रिसट्ठी', 'चौंसट्ठी', 'पैंसट्ठी', 'छयसट्ठी', 'सतसट्ठी', 'अठसट्ठी', 'उनन्सत्तरी', 'सत्तरी', 'एकहत्तर', 'बहत्तर', 'त्रिहत्तर', 'चौहत्तर', 'पचहत्तर', 'छयहत्तर', 'सतहत्तर', 'अठहत्तर', 'उनासी', 'असी', 'एकासी', 'बयासी', 'त्रियासी', 'चौरासी', 'पचासी', 'छयासी', 'सतासी', 'अठासी', 'उनान्नब्बे', 'नब्बे', 'एकान्नब्बे', 'बयान्नब्बे', 'त्रियान्नब्बे', 'चौरान्नब्बे', 'पन्चान्नब्बे', 'छयान्नब्बे', 'सन्तान्नब्बे', 'अन्ठान्नब्बे', 'उनान्सय'] + ['सय', 'हजार', 'लाख', 'करोड', 'मात्र']
    tokens = text.split()
    out = []

    for t in tokens:
        t = 'छ' if t in 'दध६ह' else t
        t = t.replace('थ', 'य')
        t = t.replace('ध', 'घ')
        match, score, _ = process.extractOne(
            t, VALID_WORDS, scorer=fuzz.ratio
        )
        out.append(match if score > 65 else '')

    return " ".join(out)

def fetch_cheque_details(image, cheque_detect_model, yolo_text, ocr_model, micr_model, language='eng'):
    cheque, center, size = rotate_using_micr(image, cheque_detect_model)
    ocr = extract_roi(cheque, center, size)
    micr = ocr.pop()
    micr_text = micr_model.predict(micr)
    serial, bank_code, *_ = parse_micr(micr_text)
    date_text = extract_date(ocr.pop(), ocr_model)
    amt = ocr_model.predict(ocr.pop(0), num_bias=3)

    output_text = []
    for i in ocr:
        # eng = padded_resize(i)

        gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        k = 0.2
        corrected = 255 * ((gray/255.0)**k)
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        blur = cv2.GaussianBlur(corrected, (5, 5), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        blur = clahe.apply(blur)

        color = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)
        enhanced = padded_resize(color, fill_color=(0,0,0))
        eng = padded_resize(i)

        result = yolo_text(enhanced, iou=0.1, verbose=False)
        #result[0].show()

        boxes = result[0].boxes
        sorted_boxes = sort_boxes_reading_order(boxes)

        text = ''
        for line in sorted_boxes:
            for box, cls in line:

                x1, y1, x2, y2 = map(int, box)

                if (y2-y1) < 25:
                    continue


                dis = eng[y1:y2, x1:x2]
                # display(dis)
                text += ocr_model.predict(dis) + ' '
        # display(eng)
        # print('Output:')
        # print(text)
        output_text.append(text.strip())
    return {'name': output_text[0], 'amount': amt, 'date': date_text,
            'words': normalize_words_eng(output_text[1]) if language=='eng' else normalize_words_nep(output_text[1]),
            'micr':micr_text, 'serial':serial, 'bank':code2name(bank_code), 
            'bank_code':bank_code[:4] if len(bank_code)>4 else bank_code}

def words_to_int(text: str) -> int:
    text = text.lower().replace("-", " ")
    words = text.split()

    units = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19
    }

    tens = {
        "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50, "sixty": 60,
        "seventy": 70, "eighty": 80, "ninety": 90
    }

    scales = {
        "hundred": 100,
        "thousand": 1_000,
        "lakh": 100_000,
        "million": 1_000_000,
        "crore": 10_000_000,
        "billion": 1_000_000_000
    }

    total = 0
    current = 0

    for word in words:
        if word in units:
            current += units[word]
        elif word in tens:
            current += tens[word]
        elif word in scales:
            total += current * scales[word]
            current = 0

    return total + current

def nepw2engw(text):
    NEP2ENG = {
        "शून्य": "zero",
        "एक": "one", "दुई": "two", "तीन": "three", "चार": "four",
        "पाँच": "five", "छ": "six", "सात": "seven", "आठ": "eight", "नौ": "nine",

        "दस": "ten", "एघार": "eleven", "बाह्र": "twelve", "तेह्र": "thirteen",
        "चौध": "fourteen", "पन्ध्र": "fifteen", "सोह्र": "sixteen",
        "सत्र": "seventeen", "अठार": "eighteen", "उन्नाइस": "nineteen",

        "बीस": "twenty", "एक्काइस": "twenty one", "बाइस": "twenty two",
        "तेइस": "twenty three", "चौबीस": "twenty four", "पच्चीस": "twenty five",
        "छब्बीस": "twenty six", "सत्ताइस": "twenty seven",
        "अठ्ठाइस": "twenty eight", "उनन्तीस": "twenty nine",

        "तीस": "thirty", "एकतीस": "thirty one", "बत्तीस": "thirty two",
        "तेत्तीस": "thirty three", "चौँतीस": "thirty four",
        "पैंतीस": "thirty five", "छत्तीस": "thirty six",
        "सैंतीस": "thirty seven", "अठतीस": "thirty eight",
        "उनन्चालीस": "thirty nine",

        "चालीस": "forty", "एकचालीस": "forty one", "बयालीस": "forty two",
        "त्रियालीस": "forty three", "चवालीस": "forty four",
        "पैंतालीस": "forty five", "छयालीस": "forty six",
        "सतचालीस": "forty seven", "अठचालीस": "forty eight",
        "उनन्चास": "forty nine",

        "पचास": "fifty", "एकाउन्न": "fifty one", "बाउन्न": "fifty two",
        "त्रिपन्न": "fifty three", "चउन्न": "fifty four",
        "पचपन्न": "fifty five", "छपन्न": "fifty six",
        "सन्ताउन्न": "fifty seven", "अन्ठाउन्न": "fifty eight",
        "उनन्साठी": "fifty nine",

        "साठी": "sixty", "एकसाठी": "sixty one", "बयसाठी": "sixty two",
        "त्रिसाठी": "sixty three", "चौंसाठी": "sixty four",
        "पैंसाठी": "sixty five", "छैंसाठी": "sixty six",
        "सतसाठी": "sixty seven", "अठसाठी": "sixty eight",
        "उनन्सत्तरी": "sixty nine",

        "सत्तरी": "seventy", "एकहत्तर": "seventy one", "बहत्तर": "seventy two",
        "त्रिहत्तर": "seventy three", "चौहत्तर": "seventy four",
        "पचहत्तर": "seventy five", "छयहत्तर": "seventy six",
        "सतहत्तर": "seventy seven", "अठहत्तर": "seventy eight",
        "उनासी": "seventy nine",

        "असी": "eighty", "एकासी": "eighty one", "बयासी": "eighty two",
        "त्रियासी": "eighty three", "चौरासी": "eighty four",
        "पचासी": "eighty five", "छयासी": "eighty six",
        "सतासी": "eighty seven", "अठासी": "eighty eight",
        "उनान्नब्बे": "eighty nine",

        "नब्बे": "ninety", "एकानब्बे": "ninety one",
        "बयानब्बे": "ninety two", "त्रियानब्बे": "ninety three",
        "चौरानब्बे": "ninety four", "पन्चानब्बे": "ninety five",
        "छयानब्बे": "ninety six", "सन्तानब्बे": "ninety seven",
        "अन्ठानब्बे": "ninety eight", "उनान्सय": "ninety nine",
        "सय": "hundred",
        "हजार": "thousand",
        "लाख": "lakh",
        "करोड": "crore",
        "मात्र": "only",
        "मात्रै": "only",
    }

    tokens = text.strip().split()
    result = []

    for token in tokens:
        if token in NEP2ENG:
            result.append(NEP2ENG[token])
    return " ".join(result)

def words2amt(text, lang):
    if lang:
        return words_to_int(nepw2engw(text))
    else:
        return words_to_int(text)
