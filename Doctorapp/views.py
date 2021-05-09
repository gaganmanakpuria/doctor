from django.shortcuts import render
import joblib
# Create your views here.
def base(request):
    return render(request,"home.html")

def cancer(request):
    return render(request,"cancer.html")

def diabeates(request):
    return render(request,"diabeates.html")
def heart(request):
    return render(request,"heart.html")
def bmi(request):
    return render(request,"bmi.html")
def kidney(request):
    return render(request,"kidney.html")
def about(request):
    return render(request,"about.html")

def result(request):
    context={}
    if "bmi" in request.GET:
        lst=['Gender', 'Height', 'Weight']
        new_lst=[]
        for i in lst:
            i=request.GET[i]
            new_lst.append(i)
        print(new_lst)
        cls=joblib.load('model_bmilr76.joblib')
        ans=cls.predict([new_lst])
        dic={0:'EXtremely Weak',1:'Weak',2:'Normal',3:'Overweight',4:'Obesity',5:'Extreme Obesity'}
        for i in ans:
            ans=dic[i]
        context['bmi']=ans


    #  diabeates 
       
    if "diabeates" in request.GET:
        lst=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
        new_lst=[]
        for i in lst:
            i=request.GET[i]
            new_lst.append(i)
        print(new_lst)
        cls=joblib.load('model_diabeates77.joblib')
        ans=cls.predict([new_lst])
        print(ans)
        dic={0:'not Survived',1:'Survived'}
        for i in ans:
            ans=dic[i]
        context['diabeates']=ans

    # cancer
    if "cancer" in request.GET:
        
        lst=['texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concavepoints_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concavepoints_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concavepoints_worst', 'symmetry_worst', 'fractal_dimension_worst']
        new_lst=[]
        for i in lst:
            i=request.GET[i]
            new_lst.append(i)
        print(new_lst)
        cls=joblib.load('model_cancerRF96.joblib')
        ans=cls.predict([new_lst])
        print(ans)
        dic={'B':'not Survived','M':'Survived'}
        for i in ans:
            ans=dic[i]
        context['cancer']=ans

    # liver 
    if "heart" in request.GET:
            
            lst=['age','sex','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
            new_lst=[]
            for i in lst:
                i=request.GET[i]
                new_lst.append(i)
            print(new_lst)
            cls=joblib.load('model_heartRF9701.joblib')
            print(cls.feature_names)
            # ans=cls.predict([new_lst])
            print(ans)
            # dic={'B':'not Survived','M':'Survived'}
            # for i in ans:
            #     ans=dic[i]
            context['cancer']=ans

  # kidney
    if "kidney" in request.GET:
        
            
            lst=['age', 'bp', 'al', 'pcc', 'bgr', 'bu', 'sc', 'hemo', 'pcv', 'htn', 'dm','appet']
            new_lst=[]
            for i in lst:
                i=request.GET[i]
                new_lst.append(i)
            print(new_lst)
            cls=joblib.load('model_kidneylr98.joblib')
            ans=cls.predict([new_lst])
            print(ans)
            dic={1.0:'chkd',0.0:'not chkd'}
            for i in ans:
                ans=dic[i]
            context['kidney']=ans  
        
        
    # print(ans)
    return render(request,"result.html",context)