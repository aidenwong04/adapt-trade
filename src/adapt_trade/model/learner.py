from river import tree, linear_model, metrics, optim
from river.utils import Rolling

class OnlineClassifier:
    def __init__(self):
        self.hatc = tree.HoeffdingAdaptiveTreeClassifier(grace_period=450,  delta=1e-7, seed=42)
        self.lr = linear_model.LogisticRegression(optim.SGD(0.1))

        self.rocauc_hatc = Rolling(metrics.ROCAUC(), window_size=500)
        self.accuracy_hatc = Rolling(metrics.Accuracy(), window_size=500)
        self.rocauc_lr = Rolling(metrics.ROCAUC(), window_size=500)
        self.accuracy_lr = Rolling(metrics.Accuracy(), window_size=500)

        self.n_trained = 0

    def predict(self,features):
        hatc_proba = self.hatc.predict_proba_one(features)
        lr_proba = self.lr.predict_proba_one(features)

        return {
            'hatc': {
                'proba': hatc_proba,
                'prediction': self.hatc.predict_one(features)
            },
            'lr': {
                'proba': lr_proba,
                'prediction': self.lr.predict_one(features)
            }
        }

    def learn(self, features, label):
        predictions = self.predict(features)
        hatc_prediction = predictions['hatc']['prediction']
        lr_prediction = predictions['lr']['prediction']

        self.accuracy_hatc.update(label, hatc_prediction)
        self.accuracy_lr.update(label, lr_prediction)
        self.rocauc_hatc.update(label, predictions['hatc']['proba'])
        self.rocauc_lr.update(label, predictions['lr']['proba'])

        self.hatc.learn_one(features, label)
        self.lr.learn_one(features, label)

        self.n_trained += 1

        return {
            'hatc': {
                'rocauc': self.rocauc_hatc.get(),
                'accuracy': self.accuracy_hatc.get()
            },
            'lr': {
                'rocauc': self.rocauc_lr.get(),
                'accuracy': self.accuracy_lr.get()
            }
        }