from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import scipy 
import warnings
warnings.filterwarnings("ignore")
import catboost, joblib, tqdm, pickle, time
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, \
 roc_auc_score, precision_recall_fscore_support, roc_curve, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import  DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV





import flask
app = Flask(__name__)

numeric_features = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', \
                        'number_emergency', 'number_inpatient', 'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'metformin', \
                        'repaglinide', 'nateglinide', 'chlorpropamide', 'glipizide', 'glyburide', 'tolazamide', 'insulin', 'change',\
                        'diabetesMed', 'health_index', 'severity_of_disease', 'number_of_changes']
    
 
scalar = dict()
for col in numeric_features : 
    file_name = 'scalers/' + str(col) + '.joblib' 
    #print(file_name) 
    scalar[col] = joblib.load(file_name)   

with open('train_columns.txt', "rb") as fp :
    train_columns = pickle.load(fp)

rf = joblib.load('rf_model.joblib')

cat_model = joblib.load('cat_model.joblib')


@app.route('/', methods = ['GET'])
def hello_world():
    return 'Hello World!'


@app.route('/index', methods = ['GET'])
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def final_function() :  
    dictt = request.form.to_dict()
    data2 = pd.DataFrame(dictt, index = [0])
    org_columns = pickle.load(open('original_column.pkl', 'rb'))
    data2 = data2.reindex(columns = org_columns, fill_value = 0)
    data = data2.copy()
    print(data.dtypes)
    data = data.astype({'admission_type_id' : 'int64', 'discharge_disposition_id' : 'int64', 'admission_source_id' : 'int64', 'time_in_hospital' : 'int64',
                        'num_lab_procedures' : 'int64', 'num_procedures' : 'int64', 'num_medications' : 'int64', 'number_outpatient' : 'int64' ,
                        'number_emergency' : 'int64', 'number_inpatient' : 'int64', 'number_diagnoses': 'int64'})
    replaceDict = {'[0-10)' : 5,
    '[10-20)' : 15,
    '[20-30)' : 25, 
    '[30-40)' : 35, 
    '[40-50)' : 45, 
    '[50-60)' : 55,
    '[60-70)' : 65, 
    '[70-80)' : 75,
    '[80-90)' : 85,
    '[90-100)' : 95}

    data['age'] = data['age'].apply(lambda x : replaceDict[x])
    
    high_frequency = ['InternalMedicine', 'Family/GeneralPractice', 'Cardiology', 'Surgery-General', 'Orthopedics', 'Orthopedics-Reconstructive', 
                 'Emergency/Trauma', 'Urology','ObstetricsandGynecology','Psychiatry','Pulmonology ','Nephrology','Radiologist']

    low_frequency = ['Surgery-PlasticwithinHeadandNeck','Psychiatry-Addictive','Proctology','Dermatology','SportsMedicine','Speech','Perinatology',\
                'Neurophysiology','Resident','Pediatrics-Hematology-Oncology','Pediatrics-EmergencyMedicine','Dentistry','DCPTEAM','Psychiatry-Child/Adolescent',\
                'Pediatrics-Pulmonology','Surgery-Pediatric','AllergyandImmunology','Pediatrics-Neurology','Anesthesiology','Pathology','Cardiology-Pediatric',\
                'Endocrinology-Metabolism','PhysicianNotFound','Surgery-Colon&Rectal','OutreachServices',\
                'Surgery-Maxillofacial','Rheumatology','Anesthesiology-Pediatric','Obstetrics','Obsterics&Gynecology-GynecologicOnco']

    pediatrics = ['Pediatrics','Pediatrics-CriticalCare','Pediatrics-EmergencyMedicine','Pediatrics-Endocrinology','Pediatrics-Hematology-Oncology',\
               'Pediatrics-Neurology','Pediatrics-Pulmonology', 'Anesthesiology-Pediatric', 'Cardiology-Pediatric', 'Surgery-Pediatric']

    psychic = ['Psychiatry-Addictive', 'Psychology', 'Psychiatry',  'Psychiatry-Child/Adolescent', 'PhysicalMedicineandRehabilitation', 'Osteopath']


    neurology = ['Neurology', 'Surgery-Neuro',  'Pediatrics-Neurology', 'Neurophysiology']


    surgery = ['Surgeon', 'Surgery-Cardiovascular', \
          'Surgery-Cardiovascular/Thoracic', 'Surgery-Colon&Rectal', 'Surgery-General', 'Surgery-Maxillofacial', \
             'Surgery-Plastic', 'Surgery-PlasticwithinHeadandNeck',  'Surgery-Thoracic',\
             'Surgery-Vascular', 'SurgicalSpecialty', 'Podiatry']
             
    ungrouped = ['Endocrinology','Gastroenterology','Gynecology','Hematology','Hematology/Oncology','Hospitalist','InfectiousDiseases',\
           'Oncology','Ophthalmology','Otolaryngology','Pulmonology','Radiology']


    missing = ['?']

    colMedical = []

    for val in data['medical_specialty'] :
        if val in pediatrics :
            colMedical.append('pediatrics')
        elif val in psychic :
            colMedical.append('psychic')
        elif val in neurology :
            colMedical.append('neurology')
        elif val in surgery :
            colMedical.append('surgery')
        elif val in high_frequency :
            colMedical.append('high_freq')
        elif val in low_frequency :
            colMedical.append('low_freq')
        elif val in ungrouped :
            colMedical.append('ungrouped')
        elif val in missing :
            colMedical.append('missing')

    data['medical_specialty'] = colMedical

    diag_1 = 414 
    diag_2 = 250
    diag_3 = 250

    data['diag_1'] = data['diag_1'].apply(lambda x : diag_1 if x == '?' else x)
    data['diag_2'] = data['diag_1'].apply(lambda x : diag_2 if x == '?' else x)
    data['diag_3'] = data['diag_3'].apply(lambda x : diag_3 if x == '?' else x)

    data['diag_1'] = data['diag_1'].apply(lambda x : 'other' if (str(x).find('V') != -1 or str(x).find('E') != -1)  
                                        else ('circulatory' if int(float(x)) in range(390, 460) or int(float(x)) == 785
                                        else     ('respiratory' if int(float(x)) in range(460, 520) or int(float(x)) == 786
                                        else     ('digestive'   if int(float(x)) in range(520, 580) or int(float(x)) == 787
                                        else     ('diabetes'    if int(float(x)) == 250
                                        else     ('injury'      if int(float(x)) in range(800, 1000)
                                        else ('musculoskeletal' if int(float(x)) in range(710, 740)
                                        else ('genitourinary'   if int(float(x)) in range(580, 630) or int(float(x)) == 788
                                        else ('neoplasms'       if int(float(x)) in range(140, 240)
                                        else ('pregnecy'        if int(float(x)) in range(630, 680)
                                        else 'other'))))))))))

    data['diag_2'] = data['diag_2'].apply(lambda x : 'other' if (str(x).find('V') != -1 or str(x).find('E') != -1)  
                                        else ('circulatory' if int(float(x)) in range(390, 460) or int(float(x)) == 785
                                        else     ('respiratory' if int(float(x)) in range(460, 520) or int(float(x)) == 786
                                        else     ('digestive'   if int(float(x)) in range(520, 580) or int(float(x)) == 787
                                        else     ('diabetes'    if int(float(x)) == 250
                                        else     ('injury'      if int(float(x)) in range(800, 1000)
                                        else ('musculoskeletal' if int(float(x)) in range(710, 740)
                                        else ('genitourinary'   if int(float(x)) in range(580, 630) or int(float(x)) == 788
                                        else ('neoplasms'       if int(float(x)) in range(140, 240)
                                        else ('pregnecy'        if int(float(x)) in range(630, 680)
                                        else 'other'))))))))))

    data['diag_3'] = data['diag_3'].apply(lambda x : 'other' if (str(x).find('V') != -1 or str(x).find('E') != -1)  
                                        else ('circulatory' if int(float(x)) in range(390, 460) or int(float(x)) == 785
                                        else     ('respiratory' if int(float(x)) in range(460, 520) or int(float(x)) == 786
                                        else     ('digestive'   if int(float(x)) in range(520, 580) or int(float(x)) == 787
                                        else     ('diabetes'    if int(float(x)) == 250
                                        else     ('injury'      if int(float(x)) in range(800, 1000)
                                        else ('musculoskeletal' if int(float(x)) in range(710, 740)
                                        else ('genitourinary'   if int(float(x)) in range(580, 630) or int(float(x)) == 788
                                        else ('neoplasms'       if int(float(x)) in range(140, 240)
                                        else ('pregnecy'        if int(float(x)) in range(630, 680)
                                        else 'other'))))))))))  
    
    data.drop(['encounter_id', 'patient_nbr'], axis = 1, inplace = True)
    data.drop(data[data.gender == 'Unknown/Invalid'].index, inplace = True)

    data['health_index'] = data.apply(lambda x:  1 / (x['number_emergency'] + x['number_inpatient'] + x['number_outpatient'])
                                  if x['number_emergency'] != 0 or x['number_inpatient'] != 0 or x['number_outpatient'] != 0
                                  else 1, axis = 1)



    total = data['time_in_hospital'].sum() + data['num_procedures'].sum() + \
                              data['num_medications'].sum() + data['num_lab_procedures'].sum() + \
                              data['number_diagnoses'].sum()

    data['severity_of_disease'] = (data['time_in_hospital'] + data['num_procedures'] + \
                              data['num_medications'] + data['num_lab_procedures'] + \
                              data['number_diagnoses']) / total

    drugList = ['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide',\
            'glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol',\
            'troglitazone','tolazamide','examide','citoglipton','insulin','glyburide-metformin','glipizide-metformin',\
            'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone']


    number_of_changes = []
    for i in range(len(data)) :
        changeCount = 0
        for col in drugList : 
            if data.iloc[i][col] in ['Down', 'Up'] :
                changeCount += 1
        number_of_changes.append(changeCount)

    data['number_of_changes'] = number_of_changes

    data.drop(['weight', 'payer_code'], axis = 1, inplace = True)

    data['discharge_disposition_id'] = data['discharge_disposition_id'].apply(lambda x : 1 if int(x) in [6, 8, 9, 13] 
                                                                           else ( 2 if int(x) in [3, 4, 5, 14, 22, 23, 24]
                                                                           else ( 10 if int(x) in [12, 15, 16, 17]
                                                                           else ( 11 if int(x) in [19, 20, 21]
                                                                           else ( 18 if int(x) in [25, 26] 
                                                                           else int(x) )))))

    data = data[~data.discharge_disposition_id.isin([11,13,14,19,20,21])]

    data['admission_type_id'] = data['admission_type_id'].apply(lambda x : 1 if int(x) in [2, 7]
                                                            else ( 5 if int(x) in [6, 8]
                                                            else int(x) ))
    
    data['admission_source_id'] = data['admission_source_id'].apply(lambda x : 1 if int(x) in [2, 3]
                                                            else ( 4 if int(x) in [5, 6, 10, 22, 25]
                                                            else ( 9 if int(x) in [15, 17, 20, 21]
                                                            else ( 11 if int(x) in [13, 14]
                                                            else int(x) ))))
    
    data['max_glu_serum'] = data['max_glu_serum'].apply(lambda x : 200 if x == '>200' 
                                                            else ( 300 if x == '>300'                                                          
                                                            else ( 100 if x == 'Norm'
                                                            else  0)))
    
    data['A1Cresult'] = data['A1Cresult'].apply(lambda x : 7 if x == '>7' 
                                                         else (8 if  x == '>8'                                                        
                                                         else ( 5 if x == 'Norm'
                                                         else  0)))
    for col in ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]:
        data[col] = data[col].apply(lambda x : 10 if x == 'Up' 
                                              else ( -10 if x == 'Down'                                                          
                                              else ( 0 if x == 'Steady'
                                              else  -20)))


    data['change'] = data['change'].apply(lambda x : 1 if x == 'Ch'
                                                 else -1)


    data['diabetesMed'] = data['diabetesMed'].apply(lambda x : -1 if x == 'No'
                                                else 1)
    
    race = data['race'] 
    data.drop(['race'], axis = 1, inplace = True)


    rejected_features = ['gender', 'glimepiride', 'acetohexamide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', \
                         'acarbose', 'miglitol', 'troglitazone', 'examide', 'citoglipton', 'glyburide-metformin', \
                         'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
    
    cat_new = ['admission_type_id', 'readmitted','discharge_disposition_id', 'admission_source_id', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3']
    data.drop(rejected_features, axis = 1, inplace=True)
    data3 = data.copy()
    data3 = pd.get_dummies(data3, columns = cat_new)
    data3 = data3.reindex(columns = train_columns, fill_value=0)

    print("#### DONE TILL HERE #### ")
    for col in numeric_features : 
        col_train = scalar[col].transform(np.array(data3[col]).reshape(-1, 1))
        data3[col] = col_train.flatten()

    race = rf.predict(data3)
    data['race'] = race
    data.drop(['readmitted'], axis = 1, inplace = True)
    cat_new.remove('readmitted')
    data = data[cat_new + numeric_features]
    predicted = cat_model.predict(data)
    print(predicted, type(predicted))
    return flask.render_template('show_results.html', pred = int(predicted))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
