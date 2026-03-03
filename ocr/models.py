from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class BankAccount(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='bank_account')
    bank_code = models.IntegerField(default=801)
    account_no = models.CharField(max_length=20, unique=True)
    nep_name = models.CharField(max_length=20, null=True, blank=True)
    phone = models.CharField(max_length=10, unique=True)
    balance = models.FloatField()

    signature = models.ImageField(upload_to='signatures/', blank=True, null=True)

    def __str__(self):
        return f'{self.user.first_name} {self.user.last_name} - {self.account_no}'

class Statements(models.Model):
    account = models.ForeignKey(BankAccount, on_delete=models.PROTECT, related_name='statements')
    amount = models.FloatField()
    date = models.DateTimeField(auto_now_add=True)
    remarks = models.CharField(max_length=50, null=True, blank=True)

class Cheque(models.Model):
    STATUS_CHOICES = (
        ("unused", "Unused"),
        ("used", "Used"),
        ("cancelled", "Cancelled"),
        ("bounced", "Bounced"),
    )
    account = models.ForeignKey(BankAccount, on_delete=models.PROTECT, related_name='cheques')
    number = models.PositiveBigIntegerField(unique=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='unused')
    issued_at = models.DateTimeField(auto_now_add=True)
    used_at = models.DateTimeField(null=True, blank=True)


    def __str__(self):
        return str(self.number)

class AuthToken(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='token')
    key = models.CharField(max_length=40, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
