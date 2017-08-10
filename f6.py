class f6:
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
        import test
        with open('X_test.txt', 'r')as open_file:
            X = open_file.read()
            X = X.strip()
            X = re.split('[\n ]',X)
            X.remove('')
            X = np.array(X, dtype=float)

        with open('y_test.txt', 'r')as open_file_1:
            y = open_file_1.read()
            y = y.strip()
            y = re.split('[\n]', y)
            y = np.array(y, dtype=float)

            # VarianceThreshold
            sel = VarianceThreshold(threshold=1)
            sel.fit(X, y)
            X_1 = sel.transform(X)
            scores1 = sel.variances_

            # SelectKBest
            skb = SelectKBest(chi2, k=3)
            skb.fit(X, y)
            scores2 = skb.scores_
            X_2 = skb.transform(X)

            # L1
            lsvc = LinearSVC(C=0.001, penalty="l1", dual=False)
            lsvc.fit(X, y)
            model = SelectFromModel(lsvc, prefit=True)
            X_3 = model.transform(X)

            # tree
            clf = ExtraTreesClassifier()
            clf.fit(X, y)
            model = SelectFromModel(clf, prefit=True)
            scores3 = clf.feature_importances_

            # pipline
            clf = Pipeline([
                ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"))),
                ('classification', RandomForestClassifier())
            ])
            clf.fit(X, y)

            # plot
            fig = plt.figure(1)
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            y1 = []
            y2 = []
            y3 = []
            for i in range(6):
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
                ax1.plot(x1, y1, 'ro')
                ax2.set_title("SelectKBest")
                ax2.set_xlabel("features")
                ax2.set_ylabel("scores")
                ax2.plot(x2, y2, 'bo')
                ax3.set_title("ExtraTreesClassifier")
                ax3.set_title("SelectKBest")
                ax3.set_xlabel("features")
                ax3.set_ylabel("scores")
                ax3.plot(x3, y3, 'ko')
                plt.pause(1)


f6().demo()
