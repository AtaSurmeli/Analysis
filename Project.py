import warnings 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from imblearn.over_sampling import SMOTE
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, accuracy_score, r2_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.exceptions import ConvergenceWarning

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler, PowerTransformer

from sklearn.linear_model import LogisticRegression

#config
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 170)
pd.set_option("display.max_rows", None)

warnings.simplefilter(action="ignore")


numCols = []
catCols = []

def ColumnsToNumericCategoric(df, cardinalTh=20):
    columnFilterDict = {
        "carGroup": [],
        "catGroup": [],
        "numGroup": [],
    }

    for column in df.columns:
        if(df[column].dtypes == "O"):
            if(df[column].nunique() >= cardinalTh):
                columnFilterDict["carGroup"].append([df[column].name, [f"Unique Values: {df[column].nunique()}"]]) 
            elif(df[column].name != "Attrition"):
                catCols.append(df[column].name)
                columnFilterDict["catGroup"].append([df[column].name, [f"Unique Values: {df[column].nunique()}"]])
                
        elif(df[column].dtypes != "O"):
            if(df[column].nunique() <= 10): #default value <=10
                catCols.append(df[column].name)
                columnFilterDict["catGroup"].append([df[column].name, [f"Unique Values: {df[column].nunique()}"]])
            else:
                numCols.append(df[column].name)
                columnFilterDict["numGroup"].append([df[column].name, [f"Unique Values: {df[column].nunique()}"]])
    
    tempArr = [len(columnFilterDict["carGroup"]), len(columnFilterDict["catGroup"]), len(columnFilterDict["numGroup"])]
    maxLen = max(tempArr)

    for x in columnFilterDict.keys():
        columnFilterDict[x] += [""] * (maxLen - len(columnFilterDict[x]))    

    return columnFilterDict

def LabelEnco(df, column):
    le = LabelEncoder()
    le.fit(df[column]) 
    df[column] = le.transform(df[column])

def EvaluateModel(model, Xtrain_, Xtest_, yTrain_, yTest_):
    model.fit(Xtrain_, yTrain_)
    yPred = model.predict(Xtest_)
    print(confusion_matrix(yTest_, yPred))

    return classification_report(yTest_, yPred)

def Scaler(df, column):
    #scaler = PowerTransformer(method="yeo-johnson") 
    scaler = StandardScaler()
    df[column] = scaler.fit_transform(df[[column]])



orgDataFrame = pd.read_csv("HR.csv")
data = orgDataFrame.copy()

numLabels = [1, 2, 3, 4, 5, 6]


data = data.drop(["EmployeeNumber", "EmployeeCount", "MonthlyRate", "Over18"], axis=1)
ColumnsToNumericCategoric(data)


data["DistanceIncome"] = data["MonthlyIncome"] / data["DistanceFromHome"]
data["IncomeLevel_"] = data["MonthlyIncome"] / data["JobLevel"]
data["IncomeAge_"] = data["MonthlyIncome"] / data["Age"]
data["IncomeInYears_"] = data["MonthlyIncome"] / data["TotalWorkingYears"]
data["IncomeYearsAtCompany_"] = data["MonthlyIncome"] / data["YearsAtCompany"]
data["TotalYearsAgeRatio_"] =  data["Age"] / data["TotalWorkingYears"]
data["RatingSalaryHike"] =  data["PerformanceRating"] / data["PercentSalaryHike"]
data["AgeInCompany"] = data["Age"] / data["YearsAtCompany"]




#data = pd.get_dummies(data, columns=catCols,drop_first=True) COMMENTED FOR FEATURE IMPORTANCE

#LabelEnco(data, "Attrition")

for x in data.columns:
    if x not in numCols:
        LabelEnco(data, x)

for x in numCols:
    rand = np.random.randint(4, 8)
    arr = np.arange(1, rand + 1)
    data[x] = pd.cut(data[x], bins=rand, labels=arr)

for x in data.columns:
    if data[x].nunique() > 2:
        Scaler(data, x)


X = data.drop("Attrition", axis=1)
y = data["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#print(data.head())

smote = SMOTE(random_state=1)
X_trainResampled, y_trainResampled = smote.fit_resample(X_train, y_train)


logReg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=1)
print("LogisticRegression", "\n", EvaluateModel(logReg, X_train, X_test, y_train, y_test))

coeffs = logReg.coef_[0]

fi = pd.DataFrame({"Feature": X.columns, "Importance": np.abs(coeffs)})
fi = fi.sort_values("Importance", ascending=True)
fi.plot(x="Feature", y="Importance", kind="barh")

plt.show()

