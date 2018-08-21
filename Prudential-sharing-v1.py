
# coding: utf-8

# ## 1.자료 탐색
# * 라이브러리 불러오기
# * Index 설정
# * 컬럼, 로우 보여지는 갯수 설정
# * 형태 파악

# #범주형 자료가 많으면
# 통합적인 독립 변수간의 유의성
# 라소는 회귀인데 -> 피쳐 셀렉션이 없는 모델이라면, 단점을 보완해서 나온
# #cat boost 짱짱맨
# 이것 저것 해봤다
# 
# 칼럼 완전 같은 아이들 찾기
# 
# EDA의 활용
# 실제 컬럼의 관계
# 여러개의 연속형 변수인 경우 -> 하나의 연속형 변수로 만들면 좋음
# 
# 컬럼이 많아지면 로지스틱이 힘들 수도 있다.
# 
# 타겟도 1~8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
# pandas_profiling 추가
import pandas_profiling

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)


# In[3]:


life_test=pd.read_csv("test.csv",index_col="Id")


# In[4]:


life_train=pd.read_csv("train.csv",index_col="Id")


# ### 통합 데이터 만들기(미싱값 채우기 용)

# In[5]:


life_total=pd.concat([life_train,life_test], axis=0)


# In[6]:


life_total_ro=life_total.drop("Response",axis=1)


# null 값이 있는지 확인

# In[19]:


check=life_train.isnull().sum()+life_test.isnull().sum()


# In[20]:


check.sort_values(ascending=False)[lambda x : x>0]


# In[11]:


life_total.isnull().drop("Response",axis=1).sum().sort_values(ascending=False)[lambda x : x>0]


# ## 1) Pandas_profiling

# In[8]:


# 프로파일링 사용
profile=pandas_profiling.ProfileReport(life_total_ro)
# profile


# In[11]:


# 현재 폴더 위치 알아내기
import os
print(os.getcwd())


# pandas 경로 <br>
# 1. "그냥 경로/파일" 형태
# 2. "../결오/파일" 은 이전 폴더로 돌아가서 하는거

# In[ ]:


# 파일로 만들기
profile.to_file(outputfile="tmp/total.html")


# In[9]:


# 상관계수가 너무 높은 항목들 
rejected_variables = profile.get_rejected_variables(threshold=0.9)
rejected_variables


# ## 2) 전체형태

# In[89]:


life_total_ro.head()


# In[90]:


life_total_ro.describe()


# In[100]:


# 전반적 정보 확인
life_total_ro.info()


# In[99]:


# 전체 row, column수 확인
life_total_ro.shape


# In[98]:


# Null값 column 확인
len(life_total_ro.isnull().any()[lambda x: x==True])
len(life_total_ro.isnull().sum()[lambda x : x>0])


# In[92]:


#Index 중복 없음확인
len(life_total_ro.index.unique()) == len(life_total_ro.index)


# ## 3) 변수별 형태

# 각 변수의 형태 파악
# - 변수이름
# - 변수 고유값
# - 고유값 수
# - 변수 형태

# In[21]:


def variable_type(x):
    if x in ['Product_Info_1','Product_Info_2','Product_Info_3','Product_Info_5','Product_Info_6','Product_Info_7','Employment_Info_2','Employment_Info_3','Employment_Info_5','InsuredInfo_1','InsuredInfo_2','InsuredInfo_3','InsuredInfo_4','InsuredInfo_5','InsuredInfo_6','InsuredInfo_7','Insurance_History_1','Insurance_History_2','Insurance_History_3','Insurance_History_4','Insurance_History_7','Insurance_History_8','Insurance_History_9','Family_Hist_1','Medical_History_2','Medical_History_3','Medical_History_4','Medical_History_5','Medical_History_6','Medical_History_7','Medical_History_8','Medical_History_9','Medical_History_11','Medical_History_12','Medical_History_13','Medical_History_14','Medical_History_16','Medical_History_17','Medical_History_18','Medical_History_19','Medical_History_20','Medical_History_21','Medical_History_22','Medical_History_23','Medical_History_25','Medical_History_26','Medical_History_27','Medical_History_28','Medical_History_29','Medical_History_30','Medical_History_31','Medical_History_33','Medical_History_34','Medical_History_35','Medical_History_36','Medical_History_37','Medical_History_38','Medical_History_39','Medical_History_40','Medical_History_41']:
        return "nominal"
    elif x in ['Product_Info_4','Ins_Age','Ht','Wt','BMI','Employment_Info_1','Employment_Info_4','Employment_Info_6','Insurance_History_5','Family_Hist_2','Family_Hist_3','Family_Hist_4','Family_Hist_5']:
        return "contiuous"
    elif x in ['Medical_History_1','Medical_History_10','Medical_History_15','Medical_History_24','Medical_History_32']:
        return "discrete"
    else:
        return "nominal(Dummy)"
# variable_type("Product_Info_1")


# In[22]:


def eda(n) :
    uni_col=[]
    for i in n.columns:
        uni_col+=[[n["%s" %i].name, list(n["%s" %i].sort_values().unique()),
                   len(n["%s" %i].sort_values().unique()),variable_type('%s'%i)]]
    result=pd.DataFrame(data=uni_col,columns=["var_name","var_value","var_len","var_official_type"])
    # eda.to_csv("eda.csv")
    # eda.sort_values(by=['var_len'])
    return result.sort_values(by=["var_name"])


# In[16]:


eda(life_total_ro)


# 상관계수 파악(null imputation 이후 다시 해봐야 함)

# In[23]:


#profile에서 걸러진 아이 -> null imputation이후 다시 해봐야 함
rejected_variables


# In[24]:


def corr_chk(df,n):
    corr = df.corr() 
    c1 = corr.abs().unstack()
#     print(c1.sort_values(ascending = False)[lambda x: (x>n) & (x!=1)])
#     print (len(c1.sort_values(ascending = False)[lambda x: (x>n) & (x!=1)]),len(c1.sort_values(ascending = False)[lambda x: (x>n) & (x!=1)][0::2]))
#     return c1.sort_values(ascending = False)[lambda x: (x>n) & (x!=1)][0::2]
    return c1.sort_values(ascending = False)[lambda x: (x>n) & (x!=1)][0::2]


# In[25]:


#여기선 빈칸이 앞에 것과 같은 거지만, 다른데서는 어떨지 모르겠음(확인해야함)
corr_chk(life_total_ro,0.8)


# 정답과의 상관계수 정도

# In[26]:


corr_response=life_total.corr()["Response"].abs().sort_values(ascending=False)
corr_response


# ## 3.NULL 값 정리<br>
# * Null 가진 컬럼 구분
# * Null Columns 제거 기준 설정
#     * 60% 이상 null일 경우
#     * unique 값이 1개
#     *  표준편차가 작은 아이
#     *  Feature selection library 사용
# * Null imputation 하는 법
#    - 범주형은 최빈값, 수치형은 예측
# ---    
# - 의견 Null 값이 70%이상 보수적으로 하는게 좋겠다.
# - #선형 회귀는 피쳐가 많으면 힘듦 -> 라소 릿지,
# -  주성분 분석
# 

# In[98]:


# Null값 column 확인
len(life_total_ro.isnull().any()[lambda x: x==True])
len(life_total_ro.isnull().sum()[lambda x : x>0])


# Null이 있는 Column 구분

# In[27]:


life_total_ro.isnull().sum().sort_values(ascending=False)[lambda x : x>0]/len(life_total_ro)
#숫자 연산하면 숫자형 자료가 되서, 함수가 안먹음


# Null 변수들 형태 확인

# In[28]:


na_col=list(life_total_ro.isnull().sum()[lambda x : x>0].index)


# In[29]:


life_total_ro[na_col].std()


# In[17]:


# 전체에 대해서 추가적으로 다시 하기
eda_2=eda(life_total_ro).assign(na_percent=list(round(life_total_ro.isnull().sum()/len(life_total_ro),2)))
eda_2.assign(std=list(life_total_ro.std())).sort_values(by="std",ascending=False)


# In[33]:


na_col


# In[30]:


eda_null=eda(life_total_ro[na_col]).assign(na_percent=list(round(life_total_ro[na_col].isnull().sum()/len(life_total_ro),2))
                                          )
eda_null.assign(std=list(life_total_ro[na_col].std())).sort_values(by="std",ascending=False)


# In[31]:


#만약 특정 값만 거의다 가지고 있는 아이 있으면 제거하기 위해
for i in na_col:
    print(life_total_ro[i].value_counts().head()/len(life_total_ro))


# # 할것 1.employment_info skewed 되있는거 -> log로 최대한 정규분포화 하기

# boosting 할때 variable formation 하는지, 해서 바꾼값을 컬럼으로 너면, 의미의 왜곡이 있을 수 있는데 이게 성능을 높인다면 사용해야 할지

# In[32]:


# 행 전체가 missing인거 찾아라
life_total_ro.isnull().sum(axis=1).sort_values(ascending=False)


# # 2. Preprocessing & Feature Selection

# ## 분기1. null 퍼센트에 따른 제거

# In[108]:


rejected_variables1=list(eda(life_total_ro[na_col]).assign(na_percent=list(round(life_total_ro[na_col].isnull().sum()/len(life_total_ro),2))
                                 )[lambda x: x.na_percent >0.6]["var_name"])
rejected_variables1


# In[23]:


life_total_rorv1=life_total_ro.drop(rejected_variables1,axis=1)


# column 제거 확인

# In[24]:


a=[]
for i in life_total_rorv1.columns:
    if i in rejected_variables1:
        a.append(i)
len(a)


# In[25]:


print(len(life_total_ro.columns),len(life_total_rorv1.columns))


# In[109]:


na_col1=list(life_total_rorv1.isnull().sum()[lambda x: x>0].index)


# ## 분기2. Missing 값 채우기

# **요약**<br>
# - discrete : 5개
# - continuous : 8개
# 
#     - Discrete은 median<br>
#     Continuous는 prediction을 사용하는 것이 좋을 것이라 생각됨

# In[111]:


var_size=life_total_rorv1[na_col1].std()
var_size.sort_values(ascending=False)
# sort_values는 변수에 너면, 문제가 조금 생김! 아래에 변수가 잘못 뽑히는 경우가 생김


# Insurance_History_5의 response와의 상관계수 0.020420

# In[ ]:


# drop할 때 column 뺄거면 axis 꼭


# ### 1)순서 Missing 성격 규명
# 1. Missing at Random (MAR)
# 2. Missing Completely at Random (MCAR)
# 3. Missing not at Random (MNAR)<br>
# 
# 1을 만족시켜야 Multiple imputaion(MICE, Alice)적용 가능

# ---
# 안됨 ㅜㅜ

# In[55]:


#little's missing completely at random test
import impyute


# In[65]:


print(impyute.__file__)


# In[112]:


na_col1


# In[87]:


life_total_rorv1["Family_Hist_4"].shape


# In[91]:


np.ndarray(shape=(79146,1), buffer=np.array(life_total_rorv1["Family_Hist_4"]))


# In[82]:


np.ndarray(life_total_rorv1["Family_Hist_4"])


# In[96]:


impyute.utils.describe(np.ndarray(shape=(79146,1
                                        ), buffer=np.array(life_total_rorv1["Family_Hist_4"])))


# In[80]:


life_total_rorv1.to_csv("./tmp/rorv1.csv")


# R로 진행해줘 지성아!

# ---
# ### 2) Imputation
# - 평균값 사용 
#     - Mean
#     - Median
#     - Mode
# - Multiple imputation(Alice, MICE)
# - Missforest
# - KNN : 이건 차원이 높아서 별로 쓸모 없을 듯

# ## 분기2. 결측값의 dummy화 진행한 후 feature selection 다시
# - n개의 missing column에 대해서 진행하여, n-1개의 dummy를 추가 생성
# - 이후 변한 상관관계 및 이용하여 다시 분석 진행
# - ?null 값 제거 전 데이터로 하는게 의미가 있을지

# In[131]:


na_col


# In[195]:


life_total_rorv1["Employment_Info_6"].isnull().head()


# In[180]:


a={}
n=0
for i in na_col1:
    n+=1
    key="{} dummy".format(i)
    value= pd.get_dummies(life_total_rorv1[i].isnull(),prefix="a").iloc[:,1]
    a[key]=value
dummy=pd.DataFrame(data=a, index=None)
dummy


# In[181]:


life_total_rorv1_dummy=pd.concat([life_total_rorv1,dummy],axis=1)


# In[182]:


life_total_rorv1_dummy.head()


# In[190]:


life_total_na1_dummy=pd.concat([life_total.drop(rejected_variables1,axis=1),dummy],axis=1)


# (컬럼 갯수 확인)

# In[189]:


len(life_total.drop(rejected_variables1,axis=1).columns)


# In[187]:


len(life_total_dummy.columns)


# dummy 추가시 상관계수 확인

# In[192]:


corr_response_dummy=life_total_na1_dummy.corr()["Response"].abs().sort_values(ascending=False)
corr_response_dummy


# In[ ]:


life


# In[18]:


import seaborn as sn


# 이것도 나중에 공부<br>
# dummy1=pd.get_dummies(life_total_rorv1[na_col1].isnull())
# dummy1=pd.get_dummies(life_total_rorv1["Employment_Info_6"].isnull(),prefix="c")

# ## 분기3. 상관계수 제거항목 
# 
# (개인적으로 null값 제거 후 fillin하고 하는게 맞는 듯)
# 
# ---
# corr 구체적 수치와 리스트 구하기

# In[196]:


def corr_chk(df,n):
    corr = df.corr() 
    c1 = corr.abs().unstack()
#     print(c1.sort_values(ascending = False)[lambda x: (x>n) & (x!=1)])
#     print (len(c1.sort_values(ascending = False)[lambda x: (x>n) & (x!=1)]),len(c1.sort_values(ascending = False)[lambda x: (x>n) & (x!=1)][0::2]))
#     return c1.sort_values(ascending = False)[lambda x: (x>n) & (x!=1)][0::2]
    return c1.sort_values(ascending = False)[lambda x: (x>n) & (x!=1)][0::2]


# In[199]:


#여기선 빈칸이 앞에 것과 같은 거지만, 다른데서는 어떨지 모르겠음(확인해야함)
corr_chk(life_total_rorv1,0.8)


# In[198]:


#여기선 빈칸이 앞에 것과 같은 거지만, 다른데서는 어떨지 모르겠음(확인해야함)
corr_chk(life_total_rorv1_dummy,0.8)


# 정답과의 상관계수 정도

# In[102]:


corr_response=life_total.drop(na_col2,axis=1).corr()["Response"].abs().sort_values(ascending=False)
corr_response


# In[200]:


rejected_variables


# rejected_variable 은 상관계수가 높은 애들 중에 정답과의 corr이 낮은 아이들이 골라지는데, 왜그런지 모르겠음<br>
# 상관계수 99%인 아이들 빼려고 하는데 정답과의 상관계수가 다름<br>
# -> null 값 때문에 계수가 영향 받았나 확인하려고 함

# In[66]:


life_total_ro[["Medical_Keyword_48","Medical_History_6","Medical_Keyword_23","Medical_History_33","Medical_History_25"
              ,"Medical_History_26"]].isnull().sum()


# null 값이 없기에, 더 적은 상관계수를 가진 변수를 제거하려함

# In[201]:


less_corr=[]
for i in [["Medical_Keyword_48","Medical_History_6"],["Medical_Keyword_23","Medical_History_33"],["Medical_History_25","Medical_History_26"]]:
    if corr_response[i[0]] < corr_response[i[1]]:
        less_corr.append(i[0])
    else:
        less_corr.append(i[1])


# In[204]:


rejected_variables


# In[205]:


rejected_variables1


# In[202]:


less_corr


# In[219]:


# 상관계수에 의한 제거 변수
rejected_variables_corr=rejected_variables+less_corr


# In[225]:


life_total_ro[rejected_variables_corr].isnull().sum()


# In[ ]:


#rejected 제거시 변화
Medical_Keyword_48         0.992918
Medical_Keyword_23        0.992875
   Medical_History_26     0.987968
Insurance_History_3      0.981345
Insurance_History_3    0.971777
Medical_History_26        0.965195
Insurance_History_9!    0.958553# 한 부분이 완전히 사라
     0.954374
Medical_History_37   Medical_Keyword_11     0.951037
Insurance_History_3  Insurance_History_4    0.949865
Insurance_History_4    0.938002
Family_Hist_2                  0.934560
Insurance_History_4      0.915989
Ins_Age                0.896203
Medical_Keyword_15   Medical_History_23     0.857073
Ins_Age              Family_Hist_2          0.856360
BMI                  Wt                     0.854896
Medical_History_19   Medical_Keyword_30     0.838616


# In[221]:


rejected_variables2=rejected_variables1+rejected_variables_corr
rejected_variables2
len(rejected_variables2)


# In[224]:


rejected_variables2=set(rejected_variables1)|set(rejected_variables_corr)
rejected_variables2=list(rejected_variables2)
rejected_variables2


# ** 요약 ** <br>
# - 연관 데이터
# 
# | 상관계수  | |연관 변수<center>|
# |---|---|---|---|---|
# | **0.95이상** | Medical_History_25  |  Medical_History_26 | Medical_History_36  |   |
# |   |  Insurance_History_3 |  Insurance_History_9 |  Insurance_History_7 |   |
# | **0.9이상** |  Medical_History_25  |  Medical_History_26 | Medical_History_36  |   |
# |   |  Insurance_History_3 |  Insurance_History_9 |  Insurance_History_7 |  Insurance_History_4 |
# | **0.8이상** |  Medical_History_25  |  Medical_History_26 | Medical_History_36  |   |
# |   |  Insurance_History_3 |  Insurance_History_9 |  Insurance_History_7 |  Insurance_History_4 |
# |   |  Family_Hist_2 | Family_Hist_4  | Ins_Age  |   |
# |   |  Medical_History_19 | Medical_Keyword_30  |   |   |
# |   | Medical_Keyword_15  |  Medical_History_23 |   |   |
# 
# 
# 

# ## 4.상관관계 분석
# - 각 변수간의 상관관계를 바탕으로

# In[ ]:


life.columns[=="product"]


# In[ ]:


life.columns[list(map(lambda x : x.split("_")[0]=="Family",life))]


# In[ ]:


null_columns


# In[ ]:


family_col=life.columns[list(map(lambda x : x.split("_")[0]=="Family",life))]


# In[ ]:


insu_col=life.columns[list(map(lambda x : x.split("_")[0]=="Insurance",life))]


# In[ ]:


conti_col=list(eda[eda.var_official_type=="contiuous"]["var_name"])


# In[ ]:


conti_var=life[conti_col]


# In[ ]:


conti_var.isnull().sum()/len(conti_var)


# In[ ]:


insu_col


# In[ ]:


len(life[conti_col].dropna())


# In[ ]:


sns.pairplot(life[conti_col].dropna())


# In[ ]:


medical_col=life.columns[list(map(lambda x : x.split("_")[0]=="Medical",life))]


# In[ ]:


len(life.columns[list(map(lambda x : x.split("_")[0]=="Medical",life))])


# 너무 커서 각 부분부터 보려고 함

# In[ ]:


# life_naout=life.drop(null_columns,axis=1)


# sns 에서 NA값이 있으면 안해줌 -> 채워서 하던지, 없애서 하던지해야해<br>
# 나는 없애고 하는게 낫다고 판단.

# In[ ]:


life[family_col]


# In[ ]:


sns.pairplot(life[["Family_Hist_1","Family_Hist_5"]].dropna())


# # 참고
# #1. 직관
# 1. Null 20% 이상 쳐내기
# 2. 값의 bias가 지나치게 작은 경우(Std)
