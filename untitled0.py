# import numpy 
# speed=[99,86,87,88,111,86,103,87,94,78,77,85,86]
# x=numpy.mean(speed)
# print(x)

# speed=[99,86,87,88,111,86,103,87,94,78,77,85,86]
# x=numpy.median(speed)
# print(x)

# from scipy import stats
# speed=[99,86,87,88,111,86,103,87,94,78,77,85,86]
# x=stats.mode(speed)
# print(x)

# speed=[99,86,87,88,111,86,103,87,94,78,77,85,86]
# x=numpy.std(speed)
# print(x)



# ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]

# x = numpy.percentile(ages, 90)
# print(x)

# x=numpy.random.uniform(0.0, 5.0, 250)
# print(x)

# import matplotlib.pyplot as plt
# x=numpy.random.uniform(0.0, 5.0, 250)
# plt.hist(x,5)
# plt.show()
# plt.savefig(sys.stdout.buffer)
# sys.stdout.flush()


# import matplotlib.pyplot as plt

# x = numpy.random.uniform(0.0, 5.0, 100000)

# plt.hist(x, 100)
# plt.show()

# x=numpy.random.normal(5.0,1.0,10000)
# plt.hist(x,100)
# plt.show()


# x=[5,7,8,7,2,17,2,9,4,11,12,9,6]
# y=[99,86,87,88,111,86,103,87,94,78,77,85,86]
# plt.scatter(x, y)
# plt.show()


# x=numpy.random.normal(5.0, 1.0, 1000)
# y=numpy.random.normal(10.0, 2.0, 1000)
# plt.scatter(x,y)
# plt.show()

# x=[5,7,8,7,2,17,2,9,4,11,12,9,6]
# y=[99,86,87,88,111,86,103,87,94,78,77,85,86]
# plt.scatter(x, y)
# plt.show()

# from scipy import stats
# x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
# y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
# slope, intercept, r, p, std_err = stats.linregress(x, y)

# def myfunc(x):
#       return slope*x+intercept
# mymodel=list(map(myfunc,x))
# plt.scatter(x,y)
# plt.plot(x,mymodel)
# plt.show() 

# from scipy import stats
# x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
# y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
# slope, intercept, r, p, std_err = stats.linregress(x, y)
# def myfunc(x):
#     return slope*x+intercept
# speed=myfunc(10)
# print(speed)

# x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
# y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
# plt.scatter(x,y)
# plt.show()

# import numpy 
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score

# x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
# y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

# mymodel=numpy.poly1d(numpy.polyfit(x,y,3))
# print(r2_score(y,mymodel(x)))

# mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
# myline = numpy.linspace(1, 22, 100)
# plt.scatter(x, y)
# plt.plot(myline, mymodel(myline))
# plt.show()


# import numpy 
# from sklearn.metrics import r2_score

# x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
# y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

# mymodel=numpy.poly1d(numpy.polyfit(x,y,3))
# speed=mymodel(17)
# print(speed)

# import pandas
# from sklearn import linear_model
# from sklearn.preprocessing import StandardScaler
# scale = StandardScaler()
# df=pandas.read_csv(".csv")
# x=df[["weight","volume"]]
# scaledX=scale.fit_transform(x)
# print(scaledX)


# import numpy
# import matplotlib .pyplot as plt
# numpy.random.seed(2)
# x=numpy.random.normal(3,1,100)
# y=numpy.random.normal(150,40,100)/x
# plt.scatter(x,y)
# plt.show()


# import numpy
# import matplotlib .pyplot as plt
# numpy.random.seed(2)
# x=numpy.random.normal(3,1,100)
# y=numpy.random.normal(150,40,100)/x

# train_x=x[:80]
# train_y=y[:80]

# test_x=x[80:]
# test_y=y[80:]
# mymodel=numpy.poly1d(numpy.polyfit(train_x,train_y,4))
# myline=numpy.linspace(0,6,100)
# plt.scatter(train_x,train_y)
# plt.plot(myline,mymodel(myline))
# plt.show()


# import matplotlib.pyplot as plt
# import numpy
# from sklearn import metrics

# actual=numpy.random.binomial(1,.9,size=1000)
# predicted=numpy.random.binomial(1,.9,size=1000)
# confusion_matrix=metrics.confusion_matrix(actual,predicted)
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

# cm_display.plot()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# x=[4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
# y=[21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
# plt.scatter(x,y)
# plt.show()

# from scipy.cluster.hierarchy import dendrogram,linkage
# x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
# y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
# data=list(zip(x,y))
# linkage_data=linkage(data,method="ward",metric="euclidean")
# dendrogram(linkage_data)
# plt.show()


# from sklearn.cluster import AgglomerativeClustering

# x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
# y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
# data=list(zip(x,y))

# hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')

# labels = hierarchical_cluster.fit_predict(data)
# plt.scatter(x, y, c=labels)
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import AgglomerativeClustering

# x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
# y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

# dararchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
# labels = hierarchical_cluster.fit_predict(data)

# plt.scatter(x, y, c=labels)
# plt.show()ta = list(zip(x, y))

# hie

# import numpy
# from sklearn import linear_model

# x=numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
# y=numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# logr=linear_model.LogisticRegression()
# logr.fit(x,y)

# predicted=logr.predict(numpy.array([3.46]).reshape(-1,1))
# print(predicted)

import numpy
from sklearn import linear_model

# X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
# y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# logr=linear_model.LogisticRegression()
# logr.fit(X, y)
# log_odds = logr.coef_
# odds=numpy.exp(log_odds)
# print(odds)

# x=numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
# y=numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# logr=linear_model.LogisticRegression()
# logr.fit(x,y)

# def logit2prob(logr, X):
#   log_odds = logr.coef_ * X + logr.intercept_
#   odds = numpy.exp(log_odds)
#   probability = odds / (1 + odds)
#   return(probability)
# print(logit2prob(logr, x))


from sklearn import datasets
from sklearn . linear_model import LogisticRegression

# iris=datasets.load_iris()
# x=iris["data"]
# y=iris["target"]
# logit=LogisticRegression(max_iter=1000)
# print(logit.fit(x, y))
# print(logit.score(x,y))

      
# iris=datasets.load_iris()
# x=iris["data"]
# y=iris["target"]

# logit=LogisticRegression(max_iter=10000)
# C=[0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
# scores=[]
# for choice in C:
#     logit.set_params(C=choice)
#     logit.fit(x,y)
#     scores.append(logit.score(x,y))
# print(scores)


# import matplotlib.pyplot as plt
# x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
# y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

# plt.scatter(x,y)
# plt.show()

      
# from sklearn.cluster import KMeans

# data = list(zip(x, y))
# inertias = []

# for i in range(1,11):
#     kmeans = KMeans(n_clusters=i)
#     kmeans.fit(data)
#     inertias.append(kmeans.inertia_)

# plt.plot(range(1,11), inertias, marker='o')
# plt.title('Elbow method')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.show()


# KMeans=KMeans(n_clusters=2)
# KMeans.fit(data)
# plt.scatter(x,y,c=KMeans.labels_)
# plt.show()


# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.tree import DecisionTreeClassifier

# data = datasets.load_wine(as_frame = True)

# X = data.data
# y = data.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 22)

# dtree = DecisionTreeClassifier(random_state = 22)
# dtree.fit(X_train,y_train)

# y_pred = dtree.predict(X_test)

# print("Train data accuracy:",accuracy_score(y_true = y_train, y_pred = dtree.predict(X_train)))
# print("Test data accuracy:",accuracy_score(y_true = y_test, y_pred = y_pred))


# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import BaggingClassifier

# data = datasets.load_wine(as_frame = True)

# X = data.data
# y = data.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 22)

# oob_model = BaggingClassifier(n_estimators = 12, oob_score = True,random_state = 22)

# oob_model.fit(X_train, y_train)

# print(oob_model.oob_score_)


# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import BaggingClassifier
# from sklearn.tree import plot_tree

# X = data.data
# y = data.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 22)

# clf = BaggingClassifier(n_estimators = 12, oob_score = True,random_state = 22)

# clf.fit(X_train, y_train)

# plt.figure(figsize=(30, 20))

# plot_tree(clf.estimators_[0], feature_names = X.columns)



# from sklearn import datasets
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import KFold, cross_val_score

# X, y = datasets.load_iris(return_X_y=True)

# clf = DecisionTreeClassifier(random_state=42)

# k_folds = KFold(n_splits = 5)

# scores = cross_val_score(clf, X, y, cv = k_folds)

# print("Cross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Number of CV Scores used in Average: ", len(scores))


# from sklearn import datasets
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import StratifiedKFold, cross_val_score

# X, y = datasets.load_iris(return_X_y=True)

# clf = DecisionTreeClassifier(random_state=42)

# sk_folds = StratifiedKFold(n_splits = 5)

# scores = cross_val_score(clf, X, y, cv = sk_folds)

# print("Cross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Number of CV Scores used in Average: ", len(scores))


# from sklearn import datasets
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import LeaveOneOut, cross_val_score

# X, y = datasets.load_iris(return_X_y=True)

# clf = DecisionTreeClassifier(random_state=42)

# loo = LeaveOneOut()

# scores = cross_val_score(clf, X, y, cv = loo)

# print("Cross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Number of CV Scores used in Average: ", len(scores))


# from sklearn import datasets
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import LeavePOut, cross_val_score

# X, y = datasets.load_iris(return_X_y=True)

# clf = DecisionTreeClassifier(random_state=42)

# lpo = LeavePOut(p=2)

# scores = cross_val_score(clf, X, y, cv = lpo)

# print("Cross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Number of CV Scores used in Average: ", len(scores))

# from sklearn import datasets
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import ShuffleSplit, cross_val_score

# X, y = datasets.load_iris(return_X_y=True)

# clf = DecisionTreeClassifier(random_state=42)

# ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits = 5)

# scores = cross_val_score(clf, X, y, cv = ss)

# print("Cross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Number of CV Scores used in Average: ", len(scores))


# import numpy as np
# from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# n = 10000
# ratio = .95
# n_0 = int((1-ratio) * n)
# n_1 = int(ratio * n)

# y = np.array([0] * n_0 + [1] * n_1)
# y_proba = np.array([1]*n)
# y_pred = y_proba > .5

# print(f'accuracy score: {accuracy_score(y, y_pred)}')
# cf_mat = confusion_matrix(y, y_pred)
# print('Confusion matrix')
# print(cf_mat)
# print(f'class 0 accuracy: {cf_mat[0][0]/n_0}')
# print(f'class 1 accuracy: {cf_mat[1][1]/n_1}')


# y_proba_2 = np.array(
#     np.random.uniform(0, .7, n_0).tolist() +
#     np.random.uniform(.3, 1, n_1).tolist()
# )
# y_pred_2 = y_proba_2 > .5

# print(f'accuracy score: {accuracy_score(y, y_pred_2)}')
# cf_mat = confusion_matrix(y, y_pred_2)
# print('Confusion matrix')
# print(cf_mat)
# print(f'class 0 accuracy: {cf_mat[0][0]/n_0}')
# print(f'class 1 accuracy: {cf_mat[1][1]/n_1}')


# import matplotlib.pyplot as plt
# def plot_roc_curve(true_y, y_prob):
#     """
#     plots the roc  curve based  of the probabilities
    
#     """
   
    
#     fpr,tpr,thresholds = roc_curve(true_y, y_prob)
#     plt.plot(fpr, tpr)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
    
# plot_roc_curve(y, y_proba)
# print(f'model 1 AUC score: {roc_auc_score(y, y_proba)}')

# plt.show()


# plot_roc_curve(y,y_proba_2)
# print(f'model 2 AUC score: {roc_auc_score(y, y_proba_2)}')

# plt.show()


# import numpy as np

# n = 10000
# y = np.array([0] * n + [1] * n)

# y_prob_1 = np.array(
#     np.random.uniform(.25, .5, n//2).tolist() +
#     np.random.uniform(.3, .7, n).tolist() +
#     np.random.uniform(.5, .75, n//2).tolist()
# )
# y_prob_2 = np.array(
#     np.random.uniform(0, .4, n//2).tolist() +
#     np.random.uniform(.3, .7, n).tolist() +
#     np.random.uniform(.6, 1, n//2).tolist()
# )

# print(f'model 1 accuracy score: {accuracy_score(y, y_prob_1>.5)}')
# print(f'model 2 accuracy score: {accuracy_score(y, y_prob_2>.5)}')

# print(f'model 1 AUC score: {roc_auc_score(y, y_prob_1)}')
# print(f'model 2 AUC score: {roc_auc_score(y, y_prob_2)}')

# plot_roc_curve(y, y_prob_1)

# plt.show()

# fpr, tpr, thresholds = roc_curve(y, y_prob_2)
# plt.plot(fpr, tpr)


# plt.show()



# import matplotlib.pyplot as plt
# x=[4, 5, 10, 4, 3, 11, 14 , 8, 10, 12]
# y=[21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
# classes=[0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

# plt.scatter(x,y,c=classes)
# plt.show()


# from sklearn.neighbors import KNeighborsClassifier

# data = list(zip(x, y))
# knn = KNeighborsClassifier(n_neighbors=1)

# knn.fit(data, classes)

# new_x=8
# new_y=21
# new_point=[(new_x,new_y)]
# prediction = knn.predict(new_point)

# plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
# plt.text(x=new_x-1.7, y=new_y-0.7, s=f"new point, class: {prediction[0]}")
# plt.show()

# knn = KNeighborsClassifier(n_neighbors=5)

# knn.fit(data, classes)

# prediction = knn.predict(new_point)

# plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
# plt.text(x=new_x-1.7, y=new_y-0.7, s=f"new point, class: {prediction[0]}")
# plt.show()





