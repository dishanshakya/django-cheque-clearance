from .models import Cheque, BankAccount
from .apps import OcrConfig
from rapidfuzz import fuzz
from datetime import datetime, timedelta
import nepali_datetime
import cv2

def eng_date(nepali_date_str):
    """
    Convert a Nepali date in Nepali digits (BS) to English/Gregorian date (AD).

    Args:
        nepali_date_str (str): Date in 'dd-mm-yyyy' format in Nepali digits, e.g., '०१-०१-२०७८'

    Returns:
        str: Corresponding English date in 'yyyy-mm-dd' format
    """
    # Mapping of Nepali digits to English digits
    nepali_digits = '०१२३४५६७८९'
    english_digits = '0123456789'

    trans_table = str.maketrans(''.join(nepali_digits), ''.join(english_digits))
    english_digit_str = nepali_date_str.translate(trans_table)

    # Split the date string into day, month, year
    day, month, year = map(int, english_digit_str.split('-'))

    # Create a Nepali date object
    nep_date = nepali_datetime.date(year, month, day)

    # Convert to English/Gregorian date
    eng_date = nep_date.to_datetime_date()

    return eng_date.strftime("%d-%m-%Y")

def is_six_months_old(date_str):
    """
    Check if the given date string (dd-mm-yyyy) is 6 months old or more from today.

    Args:
        date_str (str): Date in 'dd-mm-yyyy' format.

    Returns:
        bool: True if the date is 6 months old or more, False otherwise.
    """
    # Parse the input date string
    try:
        input_date = datetime.strptime(date_str, "%d-%m-%Y")
    except ValueError:
        raise ValueError("Date must be in 'dd-mm-yyyy' format")

    today = datetime.today()

    # Approximate 6 months as 182 days (roughly half a year)
    six_months_ago = today - timedelta(days=182)
    print('reached here')

    return input_date <= six_months_ago

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

    ref_sign = cv2.imread(bank_account.signature.path)
    test_sign1 = data.get('signa', None)
    test_sign2 = data.get('signb', None)

    if OcrConfig.scnn.predict(ref_sign, test_sign1) == 1:
        if OcrConfig.scnn.predict(ref_sign, test_sign2) == 1:
            raise Exception('Signature verification failed')

    return bank_account, cheque

def validate_drawee(user, data):
    lang = data.get('lang')
    if lang:
        user_name = user.bank_account.nep_name
        if is_six_months_old(eng_date(data.get('date'))):
            raise Exception('Cheque is older than six months')
    else:
        user_name = user.get_full_name()
        if is_six_months_old(data.get('date')):
            raise Exception('Cheque is older than six months')
    cheque_name = data.get('name', None)

    if fuzz.ratio(user_name, cheque_name) < 80:
        return False
    return True


    




