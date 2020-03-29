from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier

class classifiers:

    def __init__(self, train_features, target_train, test_features):

        self.train_features = train_features
        self.target_train = target_train
        self.test_features = test_features

    def bayesClassifier(self):

        gnb = BernoulliNB()
        gnb.fit(self.train_features, self.target_train)

        predictions = gnb.predict(self.test_features)

        return gnb, predictions

    def randomForestClassifier(self):

        clfr = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                      max_depth=None, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_decrease=0.0, min_impurity_split=None,
                                      min_samples_leaf=10, min_samples_split=2,
                                      min_weight_fraction_leaf=0.0, n_estimators=70)

        clfr.fit(self.train_features, self.target_train)

        predictions = clfr.predict(self.test_features)

        return clfr, predictions

    def deepNetworks(self):

        numpy.random.seed(7000)

        # create model
        model = Sequential()
        model.add(Dense(100, input_dim=4000, activation='relu'))
        model.add(Dense(80, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        model.fit(self.train_features, self.target_train, epochs=50, batch_size=200)

        prediction = model.predict(self.test_features)

        return model, prediction
