class f10:
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

        with open('forestfires.csv','r')as f:
            fires=f.read()
            fires=fires.strip()
            fires=re.split('[\n,]',fires)
            del fires[0:13]
            for i in range(len(fires)):
                if fires[i]=='mar':
                    fires[i]='3'
                elif fires[i]=='jan':
                    fires[i]='1'
                elif fires[i]=='sep':
                    fires[i]='9'
                elif fires[i]=='aug':
                    fires[i]='8'
                elif fires[i]=='oct':
                    fires[i]='10'
                elif fires[i]=='apr':
                    fires[i]='4'
                elif fires[i]=='jun':
                    fires[i]='6'
                elif fires[i]=='jul':
                    fires[i]='7'
                elif fires[i]=='feb':
                    fires[i]='2'
                elif fires[i]=='may':
                    fires[i]='5'
                elif fires[i]=='nov':
                    fires[i]='11'
                elif fires[i] == 'dec':
                    fires[i]='12'
                elif fires[i]=='mon':
                    fires[i]='1'
                elif fires[i]=='tue':
                    fires[i]='2'
                elif fires[i]=='wed':
                    fires[i]='3'
                elif fires[i]=='thu':
                    fires[i]='4'
                elif fires[i]=='fri':
                    fires[i]='5'
                elif fires[i]=='sat':
                    fires[i]='6'
                elif fires[i]=='sun':
                    fires[i]='7'

            fires=[fires[i:i+13] for i in range (0,len(fires),13)]
            fires=np.array(fires,dtype=float)
            X=np.delete(fires,[12],axis=1)
            y=fires.T[12]

            # feature selection
            sel = VarianceThreshold(threshold=1)
            sel.fit(X, y)
            scores1 = sel.variances_
            X_new_1 = sel.transform(X)

            '''skb = SelectKBest(chi2, k=2)
            skb.fit(X, y)
            scores2 = skb.scores_
            X_new_2 = skb.transform(X)'''

            lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
            model = SelectFromModel(lsvc, prefit=True)
            X_new_3 = model.transform(X)
            X_new_3.shape

            clf = ExtraTreesClassifier()
            clf.fit(X, y)
            model = SelectFromModel(clf, prefit=True)
            scores3 = clf.feature_importances_
            X_new_4 = model.transform(X)

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
            # 0.333333333333 0.333333333333 0.666666666667 0.333333333333



            # plot
            '''fig = plt.figure(1)
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)

            for i in range(13):
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
                ax1.set_xlim(-1, 15, 1)
                ax1.plot(x1, y1, 'ro')
                ax2.set_title("SelectKBest")
                ax2.set_xlabel("features")
                ax2.set_ylabel("scores")
                ax2.set_xlim(-1, 15, 1)
                ax2.plot(x2, y2, 'bo')
                ax3.set_title("ExtraTreesClassifier")
                ax3.set_title("SelectKBest")
                ax3.set_xlabel("features")
                ax3.set_ylabel("scores")
                ax3.set_xlim(-1, 15, 1)
                ax3.plot(x3, y3, 'ko')
                plt.pause(1)'''


f10().demo()