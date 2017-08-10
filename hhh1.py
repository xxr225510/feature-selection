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


class hhh1:
    def demo(self):
        with open('abalone.data','r')as open_file:
            abalone=open_file.read()
            abalone=abalone.strip()
            abalone=re.split('[\n ,]',abalone)
            #print abalone


            for index in range(len(abalone)):

                if abalone[index]=='M':
                    abalone[index]='0'
                elif abalone[index]=='F':
                    abalone[index]='1'
                elif abalone[index]=='I':
                    abalone[index]='2'

            abalone= [abalone[i:i + 9] for i in range(0, len(abalone), 9)]
            abalone=np.array(abalone,dtype=float)
            X= np.delete(abalone, [0], axis=1)
            y=abalone.T[0]

            # feature selection
            # VarianceThreshold
            sel = VarianceThreshold(threshold=1)
            sel.fit(X, y)
            scores1 = sel.variances_
            index1 = np.argsort(scores1)
            n = index1[:-4]
            X_new_1 = np.delete(X, [n], axis=1)

            # SelectKBest
            skb = SelectKBest(chi2, k=3)
            skb.fit(X, y)
            scores2 = skb.scores_
            index2 = np.argsort(scores2)
            n = index2[:-4]
            X_new_2 = np.delete(X, [n], axis=1)

            # L1
            lsvc = LinearSVC(C=0.043, penalty="l1", dual=False)
            lsvc.fit(X, y)
            model = SelectFromModel(lsvc, prefit=True)
            X_new_3 = lsvc.transform(X)
            scores3 = lsvc.coef_
            np.abs(scores3)
            index3 = np.argsort(scores3)

            # tree
            clf = ExtraTreesClassifier()
            clf.fit(X, y)
            model = SelectFromModel(clf, prefit=True)
            scores4 = clf.feature_importances_
            index4 = np.argsort(scores4)
            n = index4[:-4]
            X_new_4 = np.delete(X, [n], axis=1)

            # pipline
            clf = Pipeline([
                ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"))),
                ('classification', RandomForestClassifier())
            ])
            clf.fit(X, y)

            X = PolynomialFeatures(interaction_only=True).fit_transform(X_new_1).astype(float)
            clf = Perceptron(fit_intercept=False, n_iter=10, shuffle=False).fit(X_new_1, y)
            clf.predict(X_new_1)
            score1 = clf.score(X_new_1, y)
            X = PolynomialFeatures(interaction_only=True).fit_transform(X_new_2).astype(float)
            clf = Perceptron(fit_intercept=False, n_iter=10, shuffle=False).fit(X_new_2, y)
            clf.predict(X_new_2)
            score2 = clf.score(X_new_2, y)
            X = PolynomialFeatures(interaction_only=True).fit_transform(X_new_3).astype(float)
            clf = Perceptron(fit_intercept=False, n_iter=10, shuffle=False).fit(X_new_3, y)
            clf.predict(X_new_3)
            score3 = clf.score(X_new_3, y)
            X = PolynomialFeatures(interaction_only=True).fit_transform(X_new_4).astype(float)
            clf = Perceptron(fit_intercept=False, n_iter=10, shuffle=False).fit(X_new_4, y)
            clf.predict(X_new_4)
            score4 = clf.score(X_new_4, y)
            print score1, score2, score3, score4
            # 0.385683504908 0.385683504908 0.386641129998 0.531242518554
            #0.385683504908 0.385683504908 0.386641129998 0.403878381614
            #0.385683504908 0.385683504908 0.386641129998 0.456787167824
            #0.385683504908 0.385683504908 0.386641129998 0.531481924826
            #0.385683504908 0.385683504908 0.386641129998 0.427100790041



            #fig, ax = plt.subplots()
            '''fig = plt.figure(1)
            ax2 = fig.add_subplot(311)
            ax3 = fig.add_subplot(312)
            ax4=fig.add_subplot(313)

            y1=[]
            y2=[]
            y3=[]

            for i in range(8):
                x_1= np.linspace(i,8,1)
                x_2=np.linspace(i,9,1)
                x_3=np.linspace(i,10,1)


                y1=scores_1[i]
                y2=scores_2[i]
                y3=scores_3[i]'''

            '''ax.cla()
                ax.set_title("festure selection")
                ax.set_xlabel("features")
                ax.set_ylabel("scores")
                ax.set_xlim(0, 10)
                ax.grid()
                ax.plot(y1, 'r^',label='Varience')
                ax.plot(y2, 'k^' ,label='selectbeskt')
                ax.plot(y3, 'bs',label='tree')
                ax.legend(loc='best')
                

                
                ax2.set_title("Varience")
                ax3.set_title("SelectKBest")
                ax4.set_title("ExtraTreesClassifier")

                ax2.set_xlabel("features")
                ax2.set_ylabel("scores")
                ax2.set_xlim(-1,10,1)
                n1=ax2.plot(x_1,y1,'r^')

                n2=ax3.plot(x_2,y2,'k^')

                n3=ax4.plot(x_3,y3,'bs')'''



            '''ax2.legend(loc='best')
                ax3.legend(loc='best')
                ax4.legend(loc='best')



                    #if ax2.legend in fig2:

                plt.pause(1.5)'''


            '''plt.clf()
            plt.plot(scores_1,'r^')
            plt.ylabel('scores')
            plt.xlabel('features')
            plt.show()
            plt.pause(1)

            plt.close()


            fig2, ax2 = plt.subplots()
            ax2.clf()
            plt.plot(scores_2, 'k^')
            plt.ylabel('scores')
            plt.xlabel('features')
            plt.show()
            plt.pause(1)


            fig3, ax3 = plt.subplots()
            ax3.clf()
            plt.plot(scores_3, 'b^')
            plt.ylabel('scores')
            plt.xlabel('features')
            plt.show()
            plt.pause(1)
            ax3.close()
            '''


hhh1().demo()
