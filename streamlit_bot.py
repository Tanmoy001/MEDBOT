import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import hashlib
import time
training = pd.read_csv('Data/Training.csv')
testing= pd.read_csv('Data/Testing.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y

reduced_data = training.groupby(training['prognosis']).max()

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)

clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)

model=SVC()
model.fit(x_train,y_train)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

def readn(nstr):
    engine = pyttsx3.init()

    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()


severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index
def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        st.write("You should take the consultation from doctor. ")
    else:
        st.write("It might not be that bad but you should take precautions.")
def getDescription():
    global description_list
    with open('DATATWO/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)



def getSeverityDict():
    global severityDictionary
    with open('DATATWO/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('DATATWO/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)


def getInfo():
    st.title("MedBot")
    st.write(" 👋Hello,Welcome to the MedBot ")
    st.write("***Make sure you give all the symptoms answers within 10s each***")
    name = st.text_input("Enter Your Name",key=501)
    if(name):
        st.write("Hello, ", name)
    return(name)

# @st.cache(suppress_st_warning=True)
# def inp_data(i):
#     name = st.text_input("Enter yes/no", key=f"text_input_{i}")
#     if name:
#         st.write("Hello, ", name)
#     return name
@st.cache_data(experimental_allow_widgets=True)
def inp_data(i):
    name = st.text_input("Enter yes/no", key=f"text_input_{i}")
    
    if (name):
        return name
    

def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]
def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))

def inp_take():
    inp = st.text_input("Enter yes or no")
    if(inp):
        return inp

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    disease_input = st.text_input("Enter the symptom you are experiencing")

    conf_inp = ''
    if(disease_input):
        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        if conf==1:
            conf_inp = st.selectbox("Select accurate symptom", cnf_dis)
            disease_input=conf_inp
        else:
            st.write("Enter valid symptom")

    num_days = 0
    if(conf_inp != ''):
        num_days=st.number_input("Okay. From how many days ?", step=1)

    @st.cache_resource(experimental_allow_widgets=True)
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            # present_disease = print_disease(tree_.value[node])
            # # print( "You may have " +  present_disease )
            # red_cols = reduced_data.columns 
            # symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            # st.write("Are you experiencing any (check the boxes that apply)")
            # symptoms_exp=[]

            # # for syms in list(symptoms_given):
            # #     inp = ''
            # #     imp = st.radio(label=syms, options=["Yes", "No"])
            # #     st.write(imp)
            # temp = symptoms_given
            # @st.cache_data(experimental_allow_widgets=True)
            # def apply_checkbox():
            #     symptoms_temporary = list(filter(st.checkbox, temp))
            #     # st.write(symptoms_temporary)
            #     return(symptoms_temporary)

            # symptoms_temporary = apply_checkbox()

            # if(st.button("Click after selection completes", type="primary")):      # print final result if all the input's are given
            #     symptoms_exp = symptoms_temporary

            #     second_prediction=sec_predict(symptoms_exp)
            #     # print(second_prediction)
            #     calc_condition(symptoms_exp,num_days)
            #     if(present_disease[0]==second_prediction[0]):
            #         st.write("You may have ", present_disease[0])
            #         st.write(description_list[present_disease[0]])

            #         # readn(f"You may have {present_disease[0]}")
            #         # readn(f"{description_list[present_disease[0]]}")

            #     else:
            #         st.write("You may have ", present_disease[0], "or ", second_prediction[0])
            #         st.write(description_list[present_disease[0]])
            #         st.write(description_list[second_prediction[0]])

            #     precution_list=precautionDictionary[present_disease[0]]
            #     st.write("Take following measures : ")
            #     for  i,j in enumerate(precution_list):
            #         st.write(i+1,")",j)
            present_disease = print_disease(tree_.value[node])
            # print( "You may have " +  present_disease )
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            # dis_list=list(symptoms_present)
            # if len(dis_list)!=0:
            #     print("symptoms present  " + str(list(symptoms_present)))
            # print("symptoms given "  +  str(list(symptoms_given)) )
            
            # @st.cache_resource(experimental_allow_widgets=True)
            st.write("Are you experiencing any ")
            symptoms_exp=[]
            
            length=len(symptoms_given)    
            i=0
            # for syms in symptoms_given:
            count=0
            j=i
            while(length>i):    
                st.write(symptoms_given[i], "? ")
                inp = inp_data(j)  
                # time.sleep(5)
                 
                if(inp):      
                    while True:   
                        # inp = st.text_input(f"Enter 'yes' or 'no' for {symptoms_given[i]}")
                            if (inp == 'yes') | (inp == 'no'):
                                i+=1
                                j+=i
                                break
                            else:
                                st.write("Provide proper answers i.e. (yes/no) : ")
                                j+=1
                                inp = inp_data(j) 
                                time.sleep(10)
                else:
                     with st.spinner('Waiting for intput...'):
                        time.sleep(10)
                     st.warning("Invalid input.re-run it.Still we are having some suggestion for you.")
                     break
                    # You can add an optional delay or condition here if needed

                # Here, inp has a valid value ('yes' or 'no')
                # Use the value of inp as needed

                  # Move to the next symptom

                
          



            second_prediction=sec_predict(symptoms_exp)
            # print(second_prediction)
            calc_condition(symptoms_exp,num_days)
            if(present_disease[0]==second_prediction[0]):
                st.write("You may have ", present_disease[0])
                st.write(description_list[present_disease[0]])

                # readn(f"You may have {present_disease[0]}")
                # readn(f"{description_list[present_disease[0]]}")

            else:
                st.write("You may have ", present_disease[0], "or ", second_prediction[0])
                st.write(description_list[present_disease[0]])
                st.write(description_list[second_prediction[0]])

            # print(description_list[present_disease[0]])
            precution_list=precautionDictionary[present_disease[0]]
            st.write("Take following measures : ")
            for  i,j in enumerate(precution_list):
                st.write(i+1,")",j)


    if(num_days > 0):
        recurse(0,1)   
    
getSeverityDict()
getDescription()
getprecautionDict()
name = getInfo()
if(name):
    tree_to_code(clf,cols)
