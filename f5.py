class f5:
    def demo(self):
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

        with open ('wpbc1.data','r')as open_file:
            wpbc=open_file.read()
            wpbc=wpbc.strip()
            wpbc=re.split('[\n,]',wpbc)
            for i in range (len(wpbc)):
                if wpbc[i]=='N':
                    wpbc[i]='0'
                elif wpbc[i]=='R':
                    wpbc[i]='1'
                elif wpbc[i]=='?':
                    wpbc[i]='0'
            wpbc=[wpbc[i:i+35] for i in range (0,len(wpbc),35)]
            wpbc=np.array(wpbc,dtype=float)
            X=np.delete(wpbc,[0],axis=1)
            y=wpbc.T[0]

            # feature selection
            # VarianceThreshold
            sel = VarianceThreshold(threshold=1)
            sel.fit(X, y)
            scores1 = sel.variances_
            index1 = np.argsort(scores1)
            n = index1[:-6]
            X_new_1 = np.delete(X, [n], axis=1)

            # SelectKBest
            skb = SelectKBest(chi2, k=3)
            skb.fit(X, y)
            scores2 = skb.scores_
            index2 = np.argsort(scores2)
            n = index2[:-6]
            X_new_2 = np.delete(X, [n], axis=1)

            # L1
            lsvc = LinearSVC(C=0.008, penalty="l1", dual=False)
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
            n = index4[:-6]
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
            # 0.00505050505051 0.00505050505051 0.00505050505051 0.00505050505051



            # plot
        '''fig = plt.figure(1)
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)

            for i in range(34):
                y1 = scores1[i]
                y2 = scores2[i]
                y3 = scores3[i]
                x1 = np.linspace(i, 8, 1)
                x2 = np.linspace(i, 8, 1)
                x3 = np.linspace(i, 8, 1)
                plt.title('feature selection')
                ax1.set_title("Varience")
                ax1.set_xlabel("features")
                ax1.set_ylabel("scores")
                ax1.set_xlim(-1, 35, 1)
                ax1.plot(x1, y1, 'ro')
                ax2.set_title("SelectKBest")
                ax2.set_xlabel("features")
                ax2.set_ylabel("scores")
                ax2.set_xlim(-1, 35, 1)
                ax2.plot(x2, y2, 'bo')
                ax3.set_title("ExtraTreesClassifier")
                ax3.set_title("SelectKBest")
                ax3.set_xlabel("features")
                ax3.set_ylabel("scores")
                ax3.set_xlim(-1, 35, 1)
                ax3.plot(x3, y3, 'ko')
                plt.pause(1)
                '''
f5().demo()