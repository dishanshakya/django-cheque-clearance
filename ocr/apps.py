from django.apps import AppConfig
import os
from .ocr import EnglishOCR, NepaliOCR, MICROCR
from ultralytics import YOLO


class OcrConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ocr'

    engocr = None
    nepocr = None
    microcr = None
    yolo_text = None
    yolo_cheque = None

    def ready(self):
        if os.environ.get("RUN_MAIN") != "true":
            return

        print('Loading models...')
        OcrConfig.engocr = EnglishOCR('ocr/weights/english_best.pth')
        OcrConfig.nepocr = NepaliOCR('ocr/weights/nepali_best.pth')
        OcrConfig.microcr = MICROCR('ocr/weights/micr_best.pth')

        OcrConfig.yolo_text = YOLO('ocr/weights/yolo_text.pt')
        OcrConfig.yolo_cheque = YOLO('ocr/weights/yolo_cheque.pt')
        print('All models loaded!')


