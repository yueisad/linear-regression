import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep = ";")

data = data[["G1", "G2", "G3", "studytime", "failures",  "absences"]]

predict = "G3"

X = np.array(data.drop([predict], axis = 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)

#will show us what the score of the best fit is for the test data
acc = linear.score(x_test, y_test)
print(acc)

print('Coefficient: \n' , linear.coef_)
print('Intercept: \n' , linear.intercept_)

predictions = linear.predict(x_test)

# the first value should be the prediction for the score of the student based on the data (Grade, studytime, failures, absences, and final score)
for x in range (len(predictions)) :
    print(predictions[x], x_test[x], y_test[x])

#adding this comment to test the branch push
