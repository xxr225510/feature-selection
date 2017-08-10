import math
import random
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from numpy import array
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
import re
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Perceptron
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import Tkinter
import threading
import matplotlib
import matplotlib.backends.backend_tkagg


class hhh:
    def demo(self):
        with open('car.data','r')as open_file:
            car=open_file.read()
            car=car.strip()
            car=re.split('[\n,]',car)


            for i in range(len(car)):
                if car[i]=='vhigh':
                    car[i]='3'
                elif car[i]=='high':
                    car[i]='2'
                elif car[i]=='med':
                    car[i]='1'
                elif car[i] == 'low':
                    car[i] = '0'
                elif car[i]=='5more':
                    car[i]='6'
                elif car[i]=='more':
                    car[i]='5'
                elif car[i] == 'small':
                    car[i] = '0'
                elif car[i] == 'big':
                    car[i] = '2'
                elif car[i] == 'unacc':
                    car[i] = '0'
                elif car[i] == 'acc':
                    car[i] = '1'
                elif car[i] == 'good':
                    car[i] = '2'
                elif car[i] == 'vgood':
                    car[i] = '3'
            car=[car[i:i+7]for i in range(0,len(car),7)]
            car=np.array(car,dtype=int)
            X=np.delete(car, [6], axis=1)
            y=car.T[6]



            #feature selection
            #VarianceThreshold
            sel=VarianceThreshold(threshold=1)
            sel.fit(X,y)
            scores1=sel.variances_
            index1=np.argsort(scores1)
            n=index1[:-3]
            X_new_1=np.delete(X,[n],axis=1)


            #SelectKBest
            skb = SelectKBest(chi2, k=3)
            skb.fit(X, y)
            scores2 = skb.scores_
            index2=np.argsort(scores2)
            n = index2[:-3]
            X_new_2 = np.delete(X, [n], axis=1)


            #L1
            lsvc=LinearSVC(C=0.0006, penalty="l1", dual=False)
            lsvc.fit(X,y)
            model = SelectFromModel(lsvc, prefit=True)
            X_new_3=lsvc.transform(X)
            scores3=lsvc.coef_
            np.abs(scores3)
            index3=np.argsort(scores3)

            #tree
            clf = ExtraTreesClassifier()
            clf.fit(X, y)
            model = SelectFromModel(clf, prefit=True)
            scores4= clf.feature_importances_
            index4=np.argsort(scores4)
            n = index4[:-3]
            X_new_4 = np.delete(X, [n], axis=1)

            #pipline
            clf = Pipeline([
                ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"))),
                ('classification', RandomForestClassifier())
            ])
            clf.fit(X, y)


            X = PolynomialFeatures(interaction_only=True).fit_transform(X_new_1).astype(float)
            clf = Perceptron(fit_intercept=False, n_iter=10, shuffle=False).fit(X_new_1, y)
            clf.predict(X_new_1)
            score1=clf.score(X_new_1, y)
            X = PolynomialFeatures(interaction_only=True).fit_transform(X_new_2).astype(float)
            clf = Perceptron(fit_intercept=False, n_iter=10, shuffle=False).fit(X_new_2, y)
            clf.predict(X_new_2)
            score2=clf.score(X_new_2, y)
            X = PolynomialFeatures(interaction_only=True).fit_transform(X_new_3).astype(float)
            clf = Perceptron(fit_intercept=False, n_iter=10, shuffle=False).fit(X_new_3, y)
            clf.predict(X_new_3)
            score3=clf.score(X_new_3, y)
            X = PolynomialFeatures(interaction_only=True).fit_transform(X_new_4).astype(float)
            clf = Perceptron(fit_intercept=False, n_iter=10, shuffle=False).fit(X_new_4, y)
            clf.predict(X_new_4)
            score4=clf.score(X_new_4, y)
            print score1,score2,score3,score4
            #0.375578703704 0.664930555556 0.560185185185 0.664930555556





            #plot
            '''fig=plt.figure(1)
            ax1=fig.add_subplot(311)
            ax2=fig.add_subplot(312)
            ax3=fig.add_subplot(313)
            y1=[]
            y2=[]
            y3=[]
            for i in range(6):
                y1=scores1[i]
                y2=scores2[i]
                y3=scores3[i]
                x1 = np.linspace(i, 8, 1)
                x2 = np.linspace(i, 8, 1)
                x3 = np.linspace(i, 8, 1)
                plt.title('feature selection')
                ax1.set_title("Varience")
                ax1.set_xlabel("features")
                ax1.set_ylabel("scores")
                ax1.set_xlim(-1, 7, 1)
                ax1.plot(x1,y1,'ro')
                ax2.set_title("SelectKBest")
                ax2.set_xlabel("features")
                ax2.set_ylabel("scores")
                ax2.set_xlim(-1, 7, 1)
                ax2.plot(x2, y2, 'bo')
                ax3.set_title("ExtraTreesClassifier")
                ax3.set_title("SelectKBest")
                ax3.set_xlabel("features")
                ax3.set_ylabel("scores")
                ax3.set_xlim(-1, 7, 1)
                ax3.plot(x3, y3, 'ko')
                plt.pause(1)
                '''
















hhh().demo()