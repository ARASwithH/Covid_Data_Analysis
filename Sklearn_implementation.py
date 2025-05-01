import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

df = pd.read_csv('Covid_Data.csv')
df = df.drop(columns=['INTUBED', 'ICU', 'DATE_DIED'])  # deleting useless columns
df['PREGNANT'] = df['PREGNANT'].replace([97, 98, 99], 2)  # changing the 'PREGNANT' missing values to 2

# splitting dataframe according to target column
X = df.drop(columns=['CLASIFFICATION_FINAL'])
y = df['CLASIFFICATION_FINAL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)  # splittinf datas to test and train

clf = DecisionTreeClassifier(max_depth=18, random_state=42)  # making tree
clf.fit(X_train, y_train)

# calculating f1 score
y_prediction = clf.predict(X_test)
f1 = f1_score(y_test, y_prediction, average='weighted')
print('f1-score:', f1)
