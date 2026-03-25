from django.contrib import admin
from .models import BankAccount, Cheque, Statements

# Register your models here.
admin.site.register(BankAccount)
admin.site.register(Cheque)
admin.site.register(Statements)
