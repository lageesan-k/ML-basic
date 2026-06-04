import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train=pd.read_csv("data/train.csv")
test=pd.read_csv("data/test.csv")

test_ids = test["PassengerId"]  

print(train.head())
print(train.columns)


train=train.drop(["PassengerId","Name","Cabin"],axis=1)
test=test.drop(["PassengerId","Name","Cabin"],axis=1)

X=train.drop("Survived",axis=1)
Y=train["Survived"]

X=pd.get_dummies(X)
test=pd.get_dummies(test)

X, test =X.align(test, join="left", axis=1)
test=test.fillna(0)

model=RandomForestClassifier()
model.fit(X,Y)

predictions=model.predict(test)

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Survived": predictions
})

print(submission.head())

submission.to_csv("submission.csv", index=False)