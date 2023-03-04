from email import message
from django.shortcuts import render
from django.http import HttpResponse
import datetime as dt
from . import models

# Create your views here.
def index(request):
    return render(request,'login.html')
def signup(request):
    
    return render(request,'signup.html')
def signin(request):
    return render(request,'signin.html')
def signupdata(request):
    d=models.Members.objects.all()
    for i in d:
        if i.acc==request.GET['accnum'] and i.passw==request.GET['pass']: 
            return HttpResponse("already registered")
    m=models.Members(fname=request.GET['na'],lname=request.GET['lna'],acc=request.GET['accnum'], amt=0.00,date=str(dt.datetime.now()),passw=request.GET['pass'])
    m.save()
    return HttpResponse("registered successfully")
def signindata(request):
    e=models.Members.objects.all()
    global accnum,passwo,amnt,finame,laname
    for i in e:
        if  (i.acc==request.GET['baccno'] and i.passw==request.GET['pass']):
            accnum=i.acc
            passwo=i.passw
            amnt=i.amt
            finame=i.fname
            laname=i.lname
            return render(request,'signinacc.html',{'acc':i.acc,'fname':i.fname})
    return HttpResponse("incorrect")
    
def creditbutton(request):
    return render(request,'creditpage.html')
def selfCredit(request):
    return render(request,'selfcredit.html')
def creditself(request):
    n=models.Members.objects.filter(acc=accnum)
    amount=0
    for i in n:
        amount=i.amt
    m=models.Members(fname=finame,passw=passwo,lname=laname,acc=accnum,amt=amount+int(request.GET['creamt']),date=str(dt.datetime.now()))
    m.save()
    return HttpResponse('credited successfully')
def creditpage(request):
    #amnt=float(amnt)+float(request.GET['creamt'])
    actnum=request.GET['actnum']
    m=models.Members.objects.filter(acc=actnum)
    mid=0
    for i in m:
        if i.id>mid:
            n=i.amt
            pw=i.passw
            fn=i.fname
            ln=i.lname
    if(n < int(request.GET['creamt'])):
        return HttpResponse("<h1>Not enough balance</h1>")
    m=models.Members(fname=fn,passw=pw,lname=ln,acc=actnum,amt=n+int(request.GET['creamt']),date=str(dt.datetime.now()))
    m.save()
    n=models.Members.objects.filter(acc=accnum)
    amount=0
    for i in n:
        amount=i.amt
    m=models.Members(fname=finame,passw=passwo,lname=laname,acc=accnum,amt=amount-int(request.GET['creamt']),date=str(dt.datetime.now()))
    m.save()
    return HttpResponse('credited successfully')
def ministate(request):
    m=models.Members.objects.filter(acc=accnum,passw=passwo)
    context={
        'mydata':m
    }
    return render(request,'ministate.html',context)
def debitbutton(request):
    return render(request,'debitpage.html')
def debitpage(request):
    m=models.Members.objects.filter(acc=accnum,passw=passwo)
    mid=0
    for i in m:
        if i.id>mid:
            n=i.amt
    if n>=int(request.GET['debtamt']):
        m=models.Members(fname=finame,passw=passwo,lname=laname,acc=accnum,amt=n-int(request.GET['debtamt']),date=str(dt.datetime.now()))
        m.save()
        return HttpResponse("debited successfully")
    return HttpResponse('not enough balance')
    