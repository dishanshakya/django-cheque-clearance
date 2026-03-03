import uuid
from .models import AuthToken
from django.http import JsonResponse
from django.contrib.auth import authenticate

def generate_token():
    return uuid.uuid4().hex

def login_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username=username, password=password)

        if user:
            key = generate_token()
            token, created = AuthToken.objects.get_or_create(user=user, defaults={'key':key})
            if not created:
                token.key = key
                token.save()
            return JsonResponse({'token':key}, status=200)

        else:
            return JsonResponse({'msg':'Invalid Credentials'}, status=400)
    else:
        return JsonResponse({'msg': 'POST required'})

def token_required(view_func):
    def wrapper(request, *args, **kwargs):
        auth = request.headers.get('Authorization')

        if not auth or not auth.startswith('Token'):
            return JsonResponse({"error": "Unauthorized"}, status=401)
        
        key = auth.split()[1]

        try:
            token = AuthToken.objects.get(key=key)
            request.user = token.user
            request.token = token
        except AuthToken.DoesNotExist:
            return JsonResponse({"error": "Invalid token"}, status=401)
        return view_func(request, *args, **kwargs)
    return wrapper


@token_required
def logout_user(request):
    request.token.delete()
    return JsonResponse({'msg': 'Logged out'})

@token_required
def prof(request):
    return JsonResponse({'msg':'you have token'})
