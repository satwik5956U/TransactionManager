from email import message
from django.shortcuts import render
from django.http import HttpResponse
import datetime as dt
from . import models
import cv2
import numpy as np
from tensorflow import keras
# Create your views here.
emotion_model = keras.models.load_model("C:/Users/NEELAM/OneDrive/Desktop/TransactionManager/ntransaction/Scripts/myproject/myapp/emotion_recognition_model.h5")

# Define the emotions list corresponding to model output classes
emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Function to detect and recognize emotions from frames
def detect_emotion(frame):
    # Convert grayscale frame to RGB (replicate the single channel three times)
    #faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    # Preprocess the frame
    resized_frame = cv2.resize(frame_rgb, (48, 48))
    input_frame = np.expand_dims(resized_frame, axis=0)
    input_frame = input_frame.astype("float32") / 255.0

    # Predict the emotion
    predictions = emotion_model.predict(input_frame)
    predicted_class = np.argmax(predictions[0])
    predicted_emotion = emotions[predicted_class]

    return predicted_emotion
def emotion(request):
    camera = cv2.VideoCapture(0)
    #faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #faces= faceDetect.detectMultiScale(gray_frame, 1.3, 3)

        # Detect and recognize emotions
        predicted_emotion = detect_emotion(gray_frame)

        # Display the emotion on the frame
        cv2.putText(frame, predicted_emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Emotion Recognition", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    return render(request,'defdel.html',{'emo':str(predicted_emotion)})
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
    