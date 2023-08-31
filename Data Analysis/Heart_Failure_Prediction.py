import pandas as pd
import numpy as np
# # Data Collection
df = pd.read_csv("heart.csv")
df.drop('Timestamp', axis=1, inplace=True)
# # Data Preprocessing
# Handling Categorical Data
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for i in df.columns:
    if df[i].dtype == "object" and i != "HeartDisease":
        df[i] = label_encoder.fit_transform(df[i])


# Normalize the data
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaler.fit(df.iloc[:, 1:11])
df.iloc[:, 1:11] = pd.DataFrame(scaler.transform(df.iloc[:, 1:11]), index=df.index, columns=df.iloc[:, 1:11].columns)

from sklearn.feature_selection import SelectKBest, chi2
X = df.loc[:, df.columns != 'HeartDisease']
y = df[['HeartDisease']]
selector = SelectKBest(chi2, k=5)
selector.fit(X, y)
X_new = selector.transform(X)
best_features = X.columns[selector.get_support(indices=True)]

# Correlation
cor = df[best_features].corr()
import seaborn as sns
import matplotlib.pyplot as plt
# Plot correlation coefficient by a heatmap
sns.heatmap(cor, vmin=-1, vmax=1, annot=True)
plt.tight_layout()
plt.show()

# Calculating the accuracy and the time taken by the classifier
from sklearn.metrics import accuracy_score
import time
# Data Splitting for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)

Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtModel = DecisionTreeClassifier()
startTime = time.time()
# Building the model using the training data set
dtModel.fit(X_train, np.ravel(y_train))
# Evaluating the model using testing data set
dtY_pred = dtModel.predict(X_test)
dtScore = round(accuracy_score(y_test, dtY_pred), 2)
dtTime = round(time.time() - startTime, 2)
# Printing the accuracy and the time taken by the classifier
print('Accuracy using Decision Tree:', dtScore)
print('Time taken using Decision Tree:', dtTime)

import matplotlib.pyplot as plt
# z
# # Plot correlation coefficient by a heatmap
# sns.heatmap(cor, vmin=-1, vmax=1, cmap='vlag', annot=True)
# plt.tight_layout()
# plt.show()
#
# from sklearn.metrics import accuracy_score
import time

# Data Splitting for training and testing
from sklearn.model_selection import train_test_split
import time

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn import tree
#
dtModel = tree.DecisionTreeClassifier()
startTime = time.time()
from sklearn.metrics import accuracy_score
#
# # Building the model using the training data set
dtModel.fit(X_train, np.ravel(y_train))
#
# # Evaluating the model using testing data set
dtY_pred = dtModel.predict(X_test)
dtScore = round(accuracy_score(y_test, dtY_pred), 2)
dtTime = round(time.time() - startTime, 2)

# # Printing the accuracy and the time taken by the classifier
print('Accuracy using Decision Tree:', dtScore)
print('Time taken using Decision Tree:', dtTime)
#

importances = dtModel.feature_importances_
print(importances)
indices = np.argsort(importances)
# features = ['MeanHR', 'MeanSKT', 'EDA_mean_norm', 'MeanAir']
features = ["Timestamp", "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR",
            "ExerciseAngina", "Oldpeak", "ST_Slope"]
# features = ['MeanHR', 'MeanSKT', 'MeanEDA', 'MeanAir', 'IndoorHumidity', 'IndoorTemperature']
# features = ['MeanHR', 'MeanSKT', 'EDA_mean_norm', 'MeanAir', 'gender', 'age', 'BMI', 'disease']
j = 5  # top j importance
fig = plt.figure(figsize=(16, 9))
plt.barh(range(j), importances[indices][len(indices) - j:], color='lightblue', align='center')
plt.yticks(range(j), [features[i] for i in indices[len(indices) - j:]])
plt.title('Feature Importances')
plt.xlabel('Relative Importance')
plt.show()
#
#
cn = ['1', '0']
fig = plt.figure(figsize=(100, 40))
vis = tree.plot_tree(dtModel, feature_names=features, class_names=cn, max_depth=20,
                     fontsize=9, proportion=True, filled=True, rounded=True)
plt.savefig('tree_high_dpi_x2', bbox_inches='tight', dpi=250)
plt.show()
tree_rules = export_text(dtModel, max_depth=20, show_weights=True, feature_names=list(X_train.columns))
tree_rules = export_text(dtModel, max_depth=20, feature_names=list(X_train.columns))
print(tree_rules)

with open('hi_rule_x2.txt', 'a') as the_file:
    the_file.write(str(tree_rules))
    the_file.write("\n")




# Neural Networks
# from sklearn.neural_network import MLPClassifier
# nnModel = MLPClassifier()
# startTime = time.time()
# # Building the model using the training data set
# nnModel.fit(X_train, np.ravel(y_train))
# # Evaluating the model using testing data set
# nnY_pred = nnModel.predict(X_test)
# nnScore = round(accuracy_score(y_test, nnY_pred), 2)
# nnTime = round(time.time() - startTime, 2)
# # Printing the accuracy and the time taken by the classifier
# print('Accuracy using Neural Networks:', nnScore)
# print('Time taken using Neural Networks:', nnTime)



from sklearn.neural_network import MLPClassifier

# nnModel = MLPClassifier()

# max_iter = epochs.
nnModel = MLPClassifier(solver='adam', activation='identity', max_iter=500, verbose=1, learning_rate_init=.001,
                        hidden_layer_sizes=(100, 50, 25), random_state=1)
startTime = time.time()
# Building the model using the training data set
nnModel.fit(X_train, np.ravel(y_train))
# Evaluating the model using testing data set
nnY_pred = nnModel.predict(X_test)

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# from sklearn.metrics import plot_confusion_matrix

nnScore = accuracy_score(y_test, nnY_pred)
f1_score = f1_score(y_test, nnY_pred, average=None)
precision_score = precision_score(y_test, nnY_pred, average=None)
recall_score = recall_score(y_test, nnY_pred, average=None)
confusion_matrix = confusion_matrix(y_test, nnY_pred)
classification_report = classification_report(y_test, nnY_pred)

nnTime = round(time.time() - startTime, 2)

# Printing the accuracy and the time taken by the classifier
print('Accuracy using Neural Networks:', nnScore)
print("F1-score: ", f1_score)
print("Precision: ", precision_score)
print("Recall: ", recall_score)
print("Confusion_matrix: \n", confusion_matrix)
print("Classification_report: \n", classification_report)
print('Time taken using Neural Networks:', nnTime)

# fig = plot_confusion_matrix(nnModel, X_test, y_test, display_labels=nnModel.classes_)
# fig.figure_.suptitle("Confusion Matrix for Personal Comfort Dataset")
# plt.show()