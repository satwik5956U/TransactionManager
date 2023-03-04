from django.db import models

# Create your models here.
class Members(models.Model):
    fname=models.CharField(max_length=255)
    lname=models.CharField(max_length=255)
    acc=models.CharField(max_length=5)
    amt=models.DecimalField(max_digits=20,decimal_places=2)
    date=models.CharField(max_length=50)
    passw=models.CharField(max_length=16)