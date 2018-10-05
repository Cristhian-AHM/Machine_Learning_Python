# Cargando Librerias
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Cargando Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

"""
#Graficos de una Variable

# Box y whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# Histogramas
dataset.hist()
plt.show()

#Graficos Multivarables 

# scatter plot matrix
scatter_matrix(dataset)
plt.show()
"""
#Se seleccionan los valores del dataset
array = dataset.values
#Se toman los valores numericos nada mas
X = array[:,0:4]
#Se toman las etiquetas
Y = array[:,4]
#Datos para la validacion
validation = 0.20
#Semilla
seed = 7
#sklean model_selection.train_test_split
#El metodo train_test_split toma por entrada uno o dos arreglos, el size de los arreglos y una semilla para generar
#datos aleatorios para entrenar y probar
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation, random_state=seed)

scoring = 'accuracy'

#Modelos que pueden aplicar a nuestros datos
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#Se evalua cada modelo.
results = []
names = []

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
	
	
#Comparando Algoritmos
fig = plt.figure()
fig.suptitle('Comparacion de Algoritmos')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


svc = SVC()
svc.fit(X_train, Y_train)
predictions = svc.predict(X_validation)

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
