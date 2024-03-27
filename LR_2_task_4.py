def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import sklearn.metrics
from pandas import read_csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
         'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
dataset = read_csv('income_data.txt', names=names)
dataset.dropna(inplace=True)

# Розділення датасету на навчальну та контрольну вибірки
array = dataset.to_numpy()

label_encoder = []
X_encoded = np.empty(array.shape)

# Кодуємо символьні атрибути у числові
for i,item in enumerate(array[0]):
    if isinstance(item, int|float):
        X_encoded[:, i] = array[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(array[:,i])

X = X_encoded[::3, :-1].astype(int)
y = X_encoded[::3, -1].astype(int)

# Разделение X и y на обучающую и контрольную выборки
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Завантажуємо алгоритми моделі
models = {}
models['LR'] = LogisticRegression(solver='liblinear', multi_class='ovr')
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC(gamma='auto')

# оцінюємо модель на кожній ітерації
results = []
names = []

for name, model in models.items():
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_validate(model, X, y, cv=kfold, scoring=['accuracy', 'recall', 'precision', 'f1'])
    results.append(cv_results)
    names.append(name)
    print(f'\n{name}')
    print(f'Accuracy {cv_results["test_accuracy"].mean():.2f}')
    print(f'Recall {cv_results["test_recall"].mean():.2f}')
    print(f'Precision {cv_results["test_precision"].mean():.2f}')
    print(f'F1 {cv_results["test_f1"].mean():.2f}')

