from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib.auth import authenticate, login, logout
import cv2
import uuid
import requests
import numpy as np
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from .apps import OcrConfig
from .utils import fetch_cheque_details, words2amt
from django.core.cache import cache
from .models import Cheque
from .validate import validate_cheque, validate_drawee

# Create your views here.
def home(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            f = request.FILES['image']
            lang = int(request.POST['language'])
            try:
                data = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                recognition_model = OcrConfig.engocr if lang == 0 else OcrConfig.nepocr

                output = fetch_cheque_details(img,
                                              language='eng' if lang == 0 else 'nep',
                                              cheque_detect_model=OcrConfig.yolo_cheque,
                                              yolo_text=OcrConfig.yolo_text,
                                              micr_model=OcrConfig.microcr,
                                              ocr_model=recognition_model)
                bank_code = output['bank_code']
                if int(bank_code) != settings.BANK_PORT:
                    f.seek(0)
                    files = {"image":f}
                    data = {"language":lang}
                    print('request to ', 'localhost', 8000+int(bank_code))
                    response = requests.post(f'http://localhost:{8000+int(bank_code)}/api/foreign/', files=files, data=data)
                    response.raise_for_status()  # raises error for 4xx/5xx

                    foreign_data = response.json()
                    foreign_token = foreign_data.get('token', 0)
                    cache.set(foreign_token, foreign_data|{'foreign':True}, timeout=10)
                    return render(request, 'result.html', foreign_data | {'token':foreign_token})

                else:
                    token = str(uuid.uuid4())
                    cache.set(token, output|{'lang':lang,'foreign':False}, timeout=300)
                    return render(request, 'result.html', output | {'token':token})
            except Exception as e:
                raise e
                #return HttpResponse(str(e))
        else:
            bank_acc = request.user.bank_account
            print(bank_acc)
            return render(request, 'home.html', {'name':bank_acc.user.first_name,
                                                 'account_no':bank_acc.account_no, 'balance': bank_acc.balance})
    else:
        return redirect('login')


def foreign(request):
    if request.method == 'POST':
        f = request.FILES['image']
        lang = int(request.POST['language'])
        try:
            data = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            recognition_model = OcrConfig.engocr if lang == 0 else OcrConfig.nepocr

            output = fetch_cheque_details(img,
                                          language='eng' if lang == 0 else 'nep',
                                          cheque_detect_model=OcrConfig.yolo_cheque,
                                          yolo_text=OcrConfig.yolo_text,
                                          micr_model=OcrConfig.microcr,
                                          ocr_model=recognition_model)
            for field in output:
                if not field:
                    return JsonResponse({'msg':f'{field} is empty'})
            bank_code = output['bank_code']
            if int(bank_code) != settings.BANK_PORT:
                print('request to ', 'localhost', 8000+int(bank_code))

            token = str(uuid.uuid4())
            cache.set(token, output|{'lang':lang, 'foreign':False, 'intrabank':False}, timeout=300)

            return  JsonResponse(output | {'token':token, 'lang':lang})
        except Exception as e:
            print(e)
            return JsonResponse({'msg':'error occured'}, status=400)

def foreign_confirm_view(request):
    if request.method != 'POST':
        return HttpResponseBadRequest()

    token = request.POST.get('token')
    data = cache.get(token)
    if not data:
        return HttpResponseBadRequest('Expired')
    cheque_no = int(data.get('serial'))
    lang = data['lang']
    amt = words2amt(data.get('words'), lang)

    try:
        minus_account, cheque = validate_cheque(data)
    except Exception as e:
        return JsonResponse({'msg': str(e)}, status=400)


    if minus_account.balance < amt:
        cheque.status = 'bounced'
        cheque.save()
        cache.delete(token)
        return HttpResponseBadRequest('Cheque bounced')
    else:
        minus_account.balance -= amt
        minus_account.save()
        cache.delete(token)
        return HttpResponse('success', status=200)

def confirm_view(request):
    if request.method != 'POST':
        return HttpResponseBadRequest()

    token = request.POST.get('token')
    data = cache.get(token)

    if not data:
        return HttpResponseBadRequest('Expired')

    if not validate_drawee(request.user, data):
        return JsonResponse({'msg':'Not your cheque'}, status=400)

    cheque_no = int(data.get('serial'))
    lang = data['lang']
    amt = words2amt(data.get('words'), lang)

    if data.get('foreign', False):
        bank_code = data.get('bank_code')
        response = requests.post(f'http://localhost:{8000+int(bank_code)}/api/foreign-confirm/', data={'token':token})

        if response.status_code != 200:
            return HttpResponseBadRequest('Error happened')

    else:
        try:
            minus_account, cheque = validate_cheque(data)
        except Exception as e:
            return JsonResponse({'msg': str(e)}, status=400)

        if minus_account.balance < amt:
            cheque.status = 'bounced'
            cheque.save()
            cache.delete(token)
            return HttpResponseBadRequest('Cheque bounced')
        else:
            minus_account.balance -= amt
            minus_account.save()
            
    plus_account = request.user.bank_account
    plus_account.balance += amt
    plus_account.save()
    cache.delete(token)
    return HttpResponse('success', status=200)


def login_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, username=username, password=password)

        if user:
            login(request, user)
            return redirect('home')
    else:
        return render(request, 'login.html')

def logout_user(request):
    logout(request)
    return redirect('login')
