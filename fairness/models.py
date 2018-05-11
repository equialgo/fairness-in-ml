import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython import display
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from .helpers import p_rule
from .helpers import plot_distributions


class FairClassifier(object):

    def __init__(self, n_features, n_sensitive, lambdas):
        self.lambdas = lambdas

        clf_inputs = Input(shape=(n_features,))
        adv_inputs = Input(shape=(1,))

        clf_net = self._create_clf_net(clf_inputs)
        adv_net = self._create_adv_net(adv_inputs, n_sensitive)
        self._trainable_clf_net = self._make_trainable(clf_net)
        self._trainable_adv_net = self._make_trainable(adv_net)
        self._clf = self._compile_clf(clf_net)
        self._clf_w_adv = self._compile_clf_w_adv(clf_inputs, clf_net, adv_net)
        self._adv = self._compile_adv(clf_inputs, clf_net, adv_net, n_sensitive)
        self._val_metrics = None
        self._fairness_metrics = None
        self.predict = self._clf.predict

    def _make_trainable(self, net):
        def make_trainable(flag):
            net.trainable = flag
            for layer in net.layers:
                layer.trainable = flag
        return make_trainable

    def _create_clf_net(self, inputs):
        dense1 = Dense(32, activation='relu')(inputs)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(32, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        dense3 = Dense(32, activation='relu')(dropout2)
        dropout3 = Dropout(0.2)(dense3)
        outputs = Dense(1, activation='sigmoid', name='y')(dropout3)
        return Model(inputs=[inputs], outputs=[outputs])

    def _create_adv_net(self, inputs, n_sensitive):
        dense1 = Dense(32, activation='relu')(inputs)
        dense2 = Dense(32, activation='relu')(dense1)
        dense3 = Dense(32, activation='relu')(dense2)
        outputs = [Dense(1, activation='sigmoid')(dense3) for _ in range(n_sensitive)]
        return Model(inputs=[inputs], outputs=outputs)

    def _compile_clf(self, clf_net):
        clf = clf_net
        self._trainable_clf_net(True)
        clf.compile(loss='binary_crossentropy', optimizer='adam')
        return clf

    def _compile_clf_w_adv(self, inputs, clf_net, adv_net):
        clf_w_adv = Model(inputs=[inputs], outputs=[clf_net(inputs)]+adv_net(clf_net(inputs)))
        self._trainable_clf_net(True)
        self._trainable_adv_net(False)
        loss_weights = [1.]+[-lambda_param for lambda_param in self.lambdas]
        clf_w_adv.compile(loss=['binary_crossentropy']*(len(loss_weights)),
                          loss_weights=loss_weights,
                          optimizer='adam')
        return clf_w_adv

    def _compile_adv(self, inputs, clf_net, adv_net, n_sensitive):
        adv = Model(inputs=[inputs], outputs=adv_net(clf_net(inputs)))
        self._trainable_clf_net(False)
        self._trainable_adv_net(True)
        adv.compile(loss=['binary_crossentropy']*n_sensitive, optimizer='adam')
        return adv

    def _compute_class_weights(self, data_set):
        class_values = [0,1]
        class_weights = []
        if len(data_set.shape) == 1:
            balanced_weights = compute_class_weight('balanced', class_values, data_set)
            class_weights.append(dict(zip(class_values, balanced_weights)))
        else:
            n_attr = data_set.shape[1]
            for attr_idx in range(n_attr):
                balanced_weights = compute_class_weight('balanced', class_values,
                                                         np.array(data_set)[:,attr_idx])
                class_weights.append(dict(zip(class_values, balanced_weights)))
        return class_weights

    def _compute_target_class_weights(self, y):
        class_values = [0,1]
        balanced_weights = compute_class_weight('balanced', class_values, y)
        class_weights = {'y': dict(zip(class_values, balanced_weights))}
        return class_weights

    def pretrain(self, x, y, z, epochs=10, verbose=0):
        self._trainable_clf_net(True)
        self._clf.fit(x, y, epochs=epochs, verbose=verbose)
        self._trainable_clf_net(False)
        self._trainable_adv_net(True)
        class_weight_adv = self._compute_class_weights(z)
        self._adv.fit(x, np.hsplit(z, z.shape[1]), class_weight=class_weight_adv,
                      epochs=epochs, verbose=verbose)

    def fit(self, x, y, z, validation_data=None, T_iter=250, batch_size=128,
            weight_sensitive_classes=True, save_figs=False):
        n_sensitive = z.shape[1]
        if validation_data is not None:
            x_val, y_val, z_val = validation_data

        class_weight_adv = self._compute_class_weights(z)
        class_weight_clf_w_adv = [{0:1.,1:1.}]+class_weight_adv
        self._val_metrics = pd.DataFrame()
        self._fairness_metrics = pd.DataFrame()
        for idx in range(T_iter):
            if validation_data is not None:
                y_pred = pd.Series(self.clf.predict(x_val).ravel(), index=y_val.index)
                self._val_metrics.loc[idx, 'ROC AUC'] = roc_auc_score(y_val, y_pred)
                self._val_metrics.loc[idx, 'Accuracy'] = (accuracy_score(y_val, (y_pred>0.5))*100)
                for sensitive_attr in z_val.columns:
                    self._fairness_metrics.loc[idx, sensitive_attr] = p_rule(y_pred,
                                                                             z_val[sensitive_attr])
                display.clear_output(wait=True)
                plot_distributions(y_pred, z_val, idx+1, self._val_metrics.loc[idx],
                                   self._fairness_metrics.loc[idx])
                if save_figs:
                    plt.savefig(f'output/{idx+1:08d}.png', bbox_inches='tight')
                plt.show(plt.gcf())

            # train adverserial
            self._trainable_clf_net(False)
            self._trainable_adv_net(True)
            self._adv.fit(x, np.hsplit(z, z.shape[1]), batch_size=batch_size,
                          class_weight=class_weight_adv,
                          epochs=1, verbose=0)

            # train classifier
            self._trainable_clf_net(True)
            self._trainable_adv_net(False)
            indices = np.random.permutation(len(x))[:batch_size]
            self._clf_w_adv.train_on_batch(x.values[indices],
                                           [y.values[indices]]+np.hsplit(z.values[indices],
                                                                         n_sensitive),
                                           class_weight=class_weight_clf_w_adv)


class FairTorchClassifier:

    def __init__(self, clf, clf_criterion, clf_optimizer, adv, adv_criterion, adv_optimizer):
        self.clf = clf
        self.clf_criterion = clf_criterion
        self.clf_optimizer = clf_optimizer
        self.adv = adv
        self.adv_optimizer = adv_optimizer
        self.adv_criterion = adv_criterion

    def fit(self, data_loader, n_clf_epochs, n_adv_epochs):
        for _ in range(n_clf_epochs):
            self._pretrain_classifier(data_loader)
        for _ in range(n_adv_epochs):
            self._pretrain_adverserial(data_loader)

    def _pretrain_classifier(self, data_loader):
        for x, y, _ in data_loader:
            self.clf.zero_grad()
            p_y = self.clf(x)
            loss = self.clf_criterion(p_y, y)
            loss.backward()
            self.clf_optimizer.step()

    def _pretrain_adverserial(self, data_loader):
        for x, _, z in data_loader:
            p_y = self.clf(x).detach()
            self.adv.zero_grad()
            p_z = self.adv(p_y)
            loss = self.adv_criterion(p_z, z)
            loss.backward()
            self.adv_optimizer.step()

    def train(self, data_loader):
        # Train adversarial
        for x, y, z, w in data_loader:
            p_y = self.clf(x)
            self.adv.zero_grad()
            p_adv = self.adv(p_y)
            loss_adv = self.adv_criterion(p_adv, z)
            loss_adv.backward(retain_graph=True)
            self.adv_optimizer.step()

        # Train classifier on single batch
        x, y, z, w = list(data_loader)[0]
        self.clf.zero_grad()
        self.clf_criterion.weight = w
        clf_loss = self.clf_criterion(p_y, y) - (clf_adv_criterion(adv(p_y), z) * lambdas).sum()
        clf_loss.backward()
        clf_optimizer.step()

        return clf, adv
