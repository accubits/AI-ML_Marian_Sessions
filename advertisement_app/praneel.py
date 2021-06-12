import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



data = pd.read_csv("Data/Admission_Predict_Ver1.1.csv")


data = data.drop(["Serial No."], axis = 1)
X = data.iloc[:,:7]
y = data.iloc[:,7:]


X_train,X_test,Y_train,Y_test = train_test_split(X, y,random_state = 10,shuffle = True,test_size = 0.2)

# plt.scatter(X_train["GRE Score"],Y_train, color = "red")
# plt.xlabel("GRE Score")
# plt.ylabel("Chance of Admission")
# plt.legend(["GRE Score"])
# plt.show()
# plt.scatter(X_train["CGPA"],Y_train, color = "green")
# plt.xlabel("CGPA")
# plt.ylabel("Chance of Admission")
# plt.legend(["CGPA"])
# plt.show()


classifier = LinearRegression()
classifier.fit(X_train,Y_train)
prediction_of_Y = classifier.predict(X_test)
prediction_of_Y = np.round(prediction_of_Y, decimals = 3)


new_df = pd.DataFrame(columns=['Chance of Admit', 'Predicted chance of Admit'])
new_df['Chance of Admit'] = Y_test['Chance of Admit ']
new_df['Predicted chance of Admit'] = prediction_of_Y

Y_test_copy = Y_test.copy()
Y_test["Predicted chance of Admit"] = prediction_of_Y 

print(new_df)
print(Y_test)
exit()


Y_test = Y_test.drop(["Predicted chance of Admit"], axis = 1)
plt.scatter(X_test["GRE Score"],Y_test, color = "red")
plt.scatter(X_test["GRE Score"], prediction_of_Y, color='purple')
plt.xlabel("GRE Score")
plt.ylabel("Chance of Admission")
plt.legend(["Actual chance for GRE Score","Predicted chance for GRE Score"])
plt.show()
plt.scatter(X_test["SOP"],Y_test, color = "blue")
plt.scatter(X_test["SOP"], prediction_of_Y, color='orange')
plt.xlabel("SOP")
plt.ylabel("Chance of Admission")
plt.legend(["Actual chance for SOP","Predicted chance for SOP"])
plt.show()
print('Accuracy: {:.2f}'.format(classifier.score(X_test, Y_test)))
my_data = X_test.append(pd.Series([332, 107, 5, 4.5, 4.0, 9.34, 0], index = X_test.columns), ignore_index = True)
print(my_data[-1:])
my_chance = classifier.predict(my_data)
my_chance[-1]
list_of_records = [pd.Series([309, 90, 4, 4, 3.5, 7.14, 0], index = X_test.columns),
                   pd.Series([300, 99, 3, 3.5, 3.5, 8.09, 0], index = X_test.columns),
                   pd.Series([304, 108, 4, 4, 3.5, 7.91, 0], index = X_test.columns),
                   pd.Series([295, 113, 5, 4.5, 4, 8.76, 1], index = X_test.columns)]
user_defined = X_test.append(list_of_records, ignore_index= True)
print(user_defined[-4:])
#Checking chances of single record without appending to previous record
single_record_values = {"GRE Score" : [327], "TOEFL Score" : [95], "University Rating" : [4.0], "SOP": [3.5], "LOR" : [4.0], "CGPA": [7.96], "Research": [1]}
single_rec_df = pd.DataFrame(single_record_values, columns = ["GRE Score",  "TOEFL Score",  "University Rating",  "SOP",  "LOR",   "CGPA",  "Research"])
print(single_rec_df)

single_chance = classifier.predict(single_rec_df)
single_chance