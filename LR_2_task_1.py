def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn import model_selection

input_file = 'income_data.txt'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25_000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        if '?' in line:
            continue

        data = line[:-1].split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1

        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1
X = np.array(X)

label_encoder = []
X_encoded = np.empty(X.shape)

# Кодуємо символьні атрибути у числові
for i,item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:,i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Розділяємо dataset на навчальний та тестовий набори
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=5)

# Навчаємо класифікатор
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X_train, y_train)

y_test_pred = classifier.predict(X_test)

# Обчислення F-міри для SVМ-класифікатора
f1 = model_selection.cross_validate(classifier, X, y, scoring='f1_weighted', cv=3)
print("F1 score: " + str(round(100*f1['test_score'].mean(), 2)) + "%")

# Обчислення акуратність для SVМ-класифікатора
accuracy = model_selection.cross_validate(classifier, X, y, scoring='accuracy', cv=3)
print("Accuracy score: " + str(round(100*accuracy['test_score'].mean(), 2)) + "%")

# Обчислення повноти для SVМ-класифікатора
recall = model_selection.cross_validate(classifier, X, y, scoring='recall', cv=3)
print("Recall score: " + str(round(100*recall['test_score'].mean(), 2)) + "%")

# Обчислення точності для SVМ-класифікатора
precision = model_selection.cross_validate(classifier, X, y, scoring='precision', cv=3)
print("Precision score: " + str(round(100*precision['test_score'].mean(), 2)) + "%")

# Передбачення результату для тестової точки даних
input_data = ['38', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family',
              'White', 'Male', '0', '0', '40', 'United-States']

# Кодування тестової точки даних
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        arr = np.array(input_data[i], ndmin=1)
        input_data_encoded[i] = label_encoder[count].transform(arr)[0]
        count += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

# Використання класифікатора для кодованої точки даних
# та виведення результату
predicted_class = classifier.predict(input_data_encoded)
print(label_encoder[-1].inverse_transform(predicted_class))