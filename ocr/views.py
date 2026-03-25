from django.conf import settings
from django.contrib.auth import authenticate
import cv2
import uuid
import requests
import numpy as np
from django.http import JsonResponse
from .apps import OcrConfig
from .utils import fetch_cheque_details, words2amt, code2name
from django.core.cache import cache
from .models import Cheque, AuthToken, BankAccount, Statements
from .validate import validate_cheque, validate_drawee
import secrets
from .auth import token_required

# Create your views here.
@token_required
def account(request):
    try:
        account = BankAccount.objects.get(user=request.user)
        return JsonResponse({
            'name': account.user.get_full_name(),
            'balance': account.balance,
            'account_no': account.account_no,
            'statements': list(account.statements.order_by('-date').values('date', 'amount', 'remarks', 'id'))

        })
    except BankAccount.DoesNotExist:
        return JsonResponse({'msg': 'Account does not exist'}, status=400)

@token_required
def cheque_upload(request):
    if True or request.user.is_authenticated:
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
                    output.pop('signa', None)
                    output.pop('signb', None)
                    return JsonResponse(output| {'token':token})
                    #return render(request, 'result.html', output | {'token':token})
            except Exception as e:
                return JsonResponse({'msg':str(e)}, status=400)
        else:
            return JsonResponse({'msg':'post required'}, status=400)
    else:
        return JsonResponse({'msg':'wrong'}, status=400)


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

            output.pop('signa', None)
            output.pop('signb', None)
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
        return JsonResponse({'msg':'Cheque bounced'}, status=400)
    else:
        minus_account.balance -= amt
        minus_account.save()
        Statements.objects.create(
            account=minus_account,
            amount=-amt, 
            remarks=f'{cheque_no} Cheque transfer to {data.name}'
        )
        cache.delete(token)
        return JsonResponse({'status':'success', 'drawer':minus_account.user.get_full_name()}, status=200)

@token_required
def confirm_view(request):
    if request.method != 'POST':
        return JsonResponse({'msg':'Post required'}, status=400)

    token = request.POST.get('token')
    data = cache.get(token)

    if not data:
        return JsonResponse({'msg':'expired'}, status=400)

    try:
        if not validate_drawee(request.user, data):
            return JsonResponse({'msg':'Not your cheque'}, status=400)
    except Exception as e:
        return JsonResponse({'msg':str(e)}, status=400)


    cheque_no = int(data.get('serial'))
    lang = data['lang']
    amt = words2amt(data.get('words'), lang)
    bank_code = data.get('bank_code')

    if data.get('foreign', False):
        try:
            response = requests.post(f'http://localhost:{8000+int(bank_code)}/api/foreign-confirm/', data={'token':token})
        except:
            return JsonResponse({'msg':'Other bank server not responding'}, status=400)


        if response.status_code != 200:
            return JsonResponse({'msg':response.json().msg}, status=400)
        else:
            drawer = response.json().drawer

    else:
        try:
            minus_account, cheque = validate_cheque(data)
            drawer = minus_account.user.get_full_name()
        except Exception as e:
            return JsonResponse({'msg': str(e)}, status=400)

        if minus_account.balance < amt:
            cheque.status = 'bounced'
            cheque.save()
            cache.delete(token)
            return JsonResponse({'msg':'Cheque bounced'}, status=400)
        else:
            minus_account.balance -= amt
            Statements.objects.create(
                account=minus_account,
                amount=-amt, 
                remarks=f'{cheque_no} Cheque transfer to {data.get('name')}'
            )
            minus_account.save()
            
    plus_account = request.user.bank_account
    plus_account.balance += amt
    plus_account.save()
    cache.delete(token)
    minus_bank = int(bank_code)

    Statements.objects.create(
        account=plus_account,
        amount=amt, 
        remarks=f'{cheque_no} Cheque deposit'
    )

    return JsonResponse({
        'status':'success', 
        'amount': amt,
        'drawer': drawer,
        'bank': code2name(str(minus_bank) if minus_bank >=1000 else '0'+str(minus_bank))
    })


