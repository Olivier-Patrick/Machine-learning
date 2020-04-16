import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

url = ('http://biostat.mc.vanderbilt.edu/'
       'wiki/pub/Main/DataSets/titanic3.xls')

df = pd.read_excel(url)
orig_df = df

df = pd.get_dummies(df, drop_first=True)
df.columns

num_cols = df.select_dtypes(include='number').columns
im = SimpleImputer()  # mean
df1 = im.fit_transform(df[num_cols])

y = df1[1]
X = df1.drop(columns='survived')

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

lr.score(X_test, y_test)

lr.predict(X.iloc[[0]])

lr.predict_proba(X.iloc[[0]])

lr.decision_function(X.iloc[[0]])