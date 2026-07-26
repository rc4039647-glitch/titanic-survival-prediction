import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay




df = pd.read_csv('titanic.csv')
print(df.info)
print(df.describe)

print(df.isnull().sum().sort_values(ascending=False))
print(df.notnull())

df ['Age']= (df['Age'].fillna(df['Age'].median()))     #fill missing value

df['Embarked'] = df["Embarked"].fillna(df['Embarked'].mode()[0])   #fill missing value

df['has_cabin'] = df['Cabin'].notnull().astype(int)
#df.dropna(subset=['Cabin'],inplace=True)      #remove Cabin column
print(df.isnull().sum().sort_values(ascending=True))

print(df.duplicated())
print(df.drop_duplicates())

print(df.groupby('Pclass')['Age'].agg(['sum','mean','max','min']))
survived_sum = df ['Survived'].sum()
print(df)

#x = [1,2,3]
#y = [10,20,30]
#plt.plot(x,y); plt.show

########### Average Fare By Passenger Class ###########

avg  = df.groupby("Pclass")["Fare"].mean().sort_index()
print(avg)

plt.figure()
plt.plot(avg.index, avg.values, marker='o', color="red")
plt.title("Average fare treand across travel class (1 = best, 3 = cheapest)")
plt.xlabel("Pclass")
plt.ylabel("Average fare")
plt.xticks([1,2,3])
plt.show()

################ Number of survivors by passenger class ###########

survived = df.groupby("Pclass")["Survived"].sum()

plt.bar(survived.astype(str), survived.values)
plt.title("Number of survivers by travel class")
plt.xlabel("Pclass")
plt.ylabel("survivors(count)")
plt.show()

############# Age Distribution of Passengers ###########3

df["Age"] = df["Age"].fillna(df['Age'].median())
plt.hist(df["Age"], bins=15, color="blue", edgecolor="white")
plt.title("Age Distributed of titanic Passengers ")
plt.xlabel("Age")
plt.ylabel("number of Passengers")
plt.show()

############ Passenger Share by Embarkation Port ##############

embarked_count = df["Embarked"].value_counts()
plt.pie(embarked_count.values, labels=embarked_count.index, autopct=lambda 
        pct: f'{pct: .1f}%\n{(int)(round(pct/100 * embarked_count.sum()))}')
plt.title("passengers share by embarkation port")
plt.show()

############# Survival Count ##############

sns.countplot(x="Survived", hue="Survived", data=df, palette=["blue", "pink"])
plt.title("count of passenger: not Survived(0),Survived(1)")
plt.xlabel("Survived")
plt.show()

############# Fare Distribution by Passenger Class ##############

sns.boxplot(x="Pclass", y="Fare", hue="Pclass", data=df)
print(df.groupby("Pclass")["Fare"].median())
plt.title("Fare Spread and outliers by travel class")
plt.show()

############# Correlation Analysis #############

numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
corr = df[numeric_cols].corr()
print(corr)

sns.heatmap(corr, annot=True, fmt=".2f",cmap="viridis")
plt.title("Correaltion heatmap -titanic numeric columns")
plt.show()

############ Set Random Seed ############

Random_State = 42
np.random.seed(Random_State)

######### Set Plot Style #############
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (7,4.5)

df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss').replace('Mme','Mrs')
df.loc[~df['Title'].isin(['Mr','Mrs','Miss','Master']), 'Title'] = 'Rare'

df['family_size'] = df['SibSp'] + df['Parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)

############## Select Features for Model ###############

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'has_cabin', 'family_size', 'is_alone', 'Title', 'Embarked']
data = df[features + ['Survived']].copy()

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

data = pd.get_dummies(data, columns=['Title', 'Embarked'], drop_first=True)
features = [c for c in data.columns if c != 'Survived']

x = data[features]
y = data['Survived']


########## Split Training and Test Data #############

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=Random_State, stratify=y)

############ Fill Missing Age using Training Median ###############

aga_median = x_train['Age'].median()
x_test['Age'] = x_test['Age'].fillna(aga_median)
x_test['Age'] = x_test['Age'].fillna(aga_median)

print("Traning rows:", x_train.shape[0])
print("Test rows:", x_test.shape[0])
print("Age median learned from TRAINING data only:", aga_median)


########### Train Logistic Regression Model #############

model = LogisticRegression(max_iter=1000, random_state=Random_State)
model.fit(x_train, y_train)

########### Model Evaluation ########

y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Not Survived","Survived"]).plot()
plt.title("Confusion Matrix")
pd.set_option("display.max_columns",None)
plt.rcParams["figure.figsize"] = (7,4)

plt.show()


########### Coefficient Interpretation #############

coef_df = pd.DataFrame({'feature': x.columns, 'coef': model.coef_[0]}).sort_values('coef', ascending=False)
print(coef_df)

coef_df = pd.DataFrame({'feature': x.columns, 'coef': model.coef_[0]}).sort_values('coef', ascending=False)
print(coef_df)

print("""
Key factors that increased survival chances:
- Being female had the strongest positive effect on survival, consistent
  with the 'women and children first' evacuation policy.
- Having a recorded cabin (has_cabin=1) was linked to higher survival,
  since it correlates with higher passenger class and proximity to lifeboats.
- Title 'Mrs' (married women) had a positive effect on survival odds.

Key factors that decreased survival chances:
- Title 'Mr' had by far the strongest negative effect, showing adult men
  had much lower survival odds than any other group.
- Higher Pclass number (3rd class) decreased survival odds, likely due to
  distance from lifeboats and lower evacuation priority.
- Being alone (is_alone=1) and larger family_size both showed negative
  effects, suggesting solo travelers and very large families struggled
  more during evacuation.

Overall, Sex and Title (a proxy for age/sex/marital status) are the two
dominant survival factors in this dataset, matching the historical account
of the 'women and children first' policy.
""")

print("Shape:", df.shape)


############## ONE HOT ENCODING ####################

df_onehot = pd.get_dummies(df,columns=["Embarked", "Pclass"], prefix=["Embarked", "Pclass"])
print(df_onehot.head())

new_cols = [c for c in df_onehot.columns if c.startswith("Embarked_") or c.startswith("Pclass_")]
print("New one-hot column creater: ", new_cols)
#df_onehot[['embark_town'] + new_cols].head()
print(df_onehot[new_cols].head())


############# NORM#############

minmax = MinMaxScaler()
df_onehot[["age_norm", "fare_norm"]] = minmax.fit_transform(df_onehot[["Age","Fare"]])
df_onehot[["Age", "age_norm", "Fare", "fare_norm"]].describe().loc[["min", "max", "mean"]]
print(minmax)


############ STANDARD SCALER #############

standard = StandardScaler()
df_onehot[["age_std", "fare_std"]] = standard.fit_transform(df_onehot[["Age","Fare"]])
#print("age_std -> mean:", round(df_onehot["a"]))
print(df_onehot[["Age", "age_std", "Fare", "fare_std"]].head())



