from .models import Cheque, BankAccount
from rapidfuzz import fuzz

def validate_cheque(data):
    cheque_no = data.get('serial', None)
    if not cheque_no:
        raise Exception('No cheque number')

    try:
        cheque = Cheque.objects.get(number=int(cheque_no))
        bank_account = cheque.account
    except:
        raise Exception('Bad Cheque')

    if cheque.status != 'unused':
        raise Exception('Cheque already used')

    return bank_account, cheque

def validate_drawee(user, data):
    lang = data.get('lang')
    if lang:
        user_name = user.bank_account.nep_name
    else:
        user_name = user.get_full_name
    cheque_name = data.get('name', None)

    if fuzz.ratio(user_name, cheque_name) < 80:
        return False
    return True

    




