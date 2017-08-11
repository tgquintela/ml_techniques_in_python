
import numpy as np
import time
import os


class Solver(object):

    def report_results(self):
        pathfile = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                'extra/templates/report_svm_template')
        with open(pathfile, 'r') as fh:
            report = fh.read()
        # The criteria of get the accuracy will be the accuracies in the epoch
        # with better test accuracy.
        if self.test_accuracy_history:
            best_i = np.argmax(self.test_accuracy_history)
            train_accuracy = self.train_accuracy_history[best_i]
            test_accuracy = self.test_accuracy_history[best_i]
            reg_pars = self.regularizer.parameters
            report = report.format(self.learning_rate, reg_pars,
                                   self.batch_size, self.fit_time,
                                   self.epoch_learned, best_i,
                                   train_accuracy, test_accuracy)
            print(report)

    def _reset_history(self):
        self.epoch_learned = 0
        self.fit_time = 0.
        self.train_loss_history = None
        self.test_loss_history = None
        self.train_accuracy_history = None
        self.test_accuracy_history = None
        if self.history:
            self.train_loss_history = []
            self.train_accuracy_history = []
            self.test_loss_history = []
            self.test_accuracy_history = []

    def _add_epoch_to_history(self, train_loss, train_accuracy,
                              test_loss=None, test_accuracy=None):
        if self.history:
            self.train_loss_history.append(train_loss)
            self.train_accuracy_history.append(train_accuracy)
            if test_loss is not None:
                self.test_loss_history.append(test_loss)
                self.test_accuracy_history.append(test_accuracy)


class SGD(Solver):
    """SVM with SGD optimizer.
    """
    def __init__(self, loss='Hinge', loss_pars=1.0, model='svm',
                 model_pars=None, regularizer='null', reg_pars=1.,
                 batch_size=10, n_epochs=0, learning_rate=0.0001,
                 stop_step=.000001, history=True, verbose=False):

        ## Definition of complentary elements
        self.lossf = LossFunction.create_lossfunction(loss, loss_pars)
        self.model = Model.create_model(model, model_pars)
        self.regularizer = Regularization.create_regularization(regularizer,
                                                                reg_pars)

        ## Optimizer parameters
        self.optimizer = 'SGD'
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.stop_step = stop_step

        ## Tracking results
        self.last_loss = 1e16
        self.change_loss = 1e16
        self.history = history
        self._reset_history()

    def predict(self, X):
        return self.model.compute(X)

    def score(self, X, y):
        """Scoring with accuracy measure by default."""
        y_pred = self.predict(X)
        return accuracy(y_pred, y)

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        """"""
        # Setting administrative parameters
        self._reset_history()
        t0 = time.time()
        # Setting general paramaters
        N_samples, n_feats = X_train.shape
        # Initialize model parameters
        self.model = self.model.weights_initialization(n_feats)
#        self._initialization_weights(n_feats)
        # Setting loop epochs
        for i in range(self.n_epochs):
            # Shuffling data
            data, labels = permut_data(X_train, y_train)
            # Loop over batches
            for x_batch, y_batch in self._batch_generator(data, labels):
                # Batch prediction
                y_batch_pred = self.predict(x_batch)
                # Update model
                self.update_model(x_batch, y_batch, y_batch_pred)

            # Tracking loss
            self.compute_epoch_measures(X_train, y_train, X_test, y_test)

            # Tracking loop
            if self.change_loss < self.stop_step:
                break

        self.fit_times = time.time()-t0
        return self

    def update_model(self, x_batch, y_batch, y_batch_pred):
        # Compute gradient
        gradloss_w, gradloss_w0 =\
            self.lossf.gradient_loss(y_batch_pred, y_batch, x_batch)
        gradloss_w, gradloss_w0 =\
            self._add_gradient_regularization(gradloss_w, gradloss_w0)
        # Parameters update
        new_w = self.model.w - self.learning_rate*gradloss_w
        new_w0 = self.model.w0 - self.learning_rate*gradloss_w0
        self.model = self.model.reinstantiate(new_w, new_w0)

    def compute_epoch_measures(self, X_train, y_train, X_test, y_test):
        ## Train
        losses_epoch, train_acc = self._compute_measures(X_train, y_train)
        # Change loss
        self.change_loss = self.last_loss-losses_epoch
        self.last_loss = losses_epoch

        ## Test
        testing_phase = (X_test is not None) and (y_test is not None)
        if testing_phase:
            loss_test, acc_test = self._compute_measures(X_test, y_test)
        else:
            loss_test, acc_test = None, None

        # Add to history
        self._add_epoch_to_history(losses_epoch, train_acc,
                                   loss_test, acc_test)

        # Add new epoch to the counter
        self.epoch_learned += 1

    def _compute_measures(self, X, y):
        # Prediction train
        y_p = self.predict(X)
        lossf_term = self.lossf.loss(y_p, y)
        loss = self._add_regularization_loss(lossf_term)

        # Accuracy train
        acc = accuracy(np.sign(y_p), y)
        return loss, acc

    def _batch_generator(self, data, labels):
        """We can implement here different options to sample batches."""
        N_samples = len(data)
        for init, endit in batch_size_iter(N_samples, self.batch_size):
            x_batch = data[init:endit]
            y_batch = labels[init:endit]
            yield x_batch, y_batch

    def _add_regularization_loss(self, loss):
        """"""
        reg_term = 0.5*(self.regularizer.regularize(*self.model.parameters))
        loss += reg_term
        return loss

    def _add_gradient_regularization(self, gradloss_w, gradloss_w0):
        grad_w_reg, grad_w0_reg =\
            self.regularizer.gradient_regularization(*self.model.parameters)
        gradloss_w += grad_w_reg
        gradloss_w0 += grad_w0_reg
        return gradloss_w, gradloss_w0

#    def _reset_model(self):
#        self.w = None
#        self.w0 = None
#
#    def _initialization_weights(self, n_feats, init_type='gauss'):
#        if init_type == 'zeros':
#            self.w0 = 0.
#            self.w = np.zeros(n_feats)
#        elif init_type == 'gauss':
#            self.w0 = np.random.randn()
#            self.w = np.random.randn(n_feats)


class SVM(SGD):

    def __init__(self, loss='Hinge', loss_pars=1.0, regularizer='l2',
                 reg_pars=1., batch_size=10, n_epochs=0,
                 learning_rate=0.0001, stop_step=.000001,
                 verbose=False, history=True):
        super(SVM, self).__init__(loss=loss, loss_pars=loss_pars,
                                  regularizer=regularizer, reg_pars=reg_pars,
                                  batch_size=batch_size, n_epochs=n_epochs,
                                  learning_rate=learning_rate,
                                  stop_step=stop_step, verbose=verbose,
                                  history=history)


def accuracy(y_pred, y_true):
    return np.sum(y_true == y_pred) / float(y_true.shape[0])


def batch_size_iter(data_size, batch_size):
    init = 0
    keep = True
    while keep:
        endit = init+batch_size
        if endit >= data_size:
            endit = data_size
            keep = False
        yield init, endit
        init += batch_size


def permut_data(data, labels=None):
    n_samples = len(data)
    reindices = np.random.permutation(n_samples)
    if labels is None:
        return data[reindices]
    else:
        return data[reindices], labels[reindices]


#def stop_condition(loss_change, i, N, stop_step):
#    if N != 0:
#        if i >= N:
#            return False
#    if loss_change < stop_step:
#        return False
#    return True


################################### Model #####################################
###############################################################################
class Model(object):

    @classmethod
    def create_model(cls, model, *parameters):
        models = {'svm': LinearModel}
        if type(model) == str:
            return models[model.lower()](*parameters)
        if isinstance(model, Model):
            return model
        else:
            return model(*parameters)

    @classmethod
    def reinstantiate(cls, *args):
        return cls(*args)


class LinearModel(Model):
    "Linar model."

    def __init__(self, w, w0=0.):
        if w is None:
            self.w = np.array([1.])
            self.w0 = 0.
        else:
            self.w = w
            self.w0 = w0

    @classmethod
    def weights_initialization(cls, n_feats, init_type='gauss'):
        if init_type == 'zeros':
            w0 = 0.
            w = np.zeros(n_feats)
        elif init_type == 'gauss':
            w = np.random.randn(n_feats)
            w0 = np.random.randn()
        return cls(w, w0)

    @property
    def parameters(self):
        return self.w, self.w0

    def compute(self, X):
        return np.dot(X, self.w)+self.w0

    def reset_model(self):
        self.w = None
        self.w0 = None


################################ Loss function ################################
###############################################################################
class LossFunction(object):
    """General object for loss functions."""

    def __init__(self):
        required_functions = ['loss', 'gradient_loss']
        for req in required_functions:
            assert(req in dir(self))

    @classmethod
    def create_lossfunction(cls, loss, *parameters):
        ## Setting loss
        loss_functions = {'hinge': Hinge}
        if type(loss) == str:
            return loss_functions[loss.lower()](*parameters)
        if isinstance(loss, LossFunction):
            return loss
        else:
            return loss(*parameters)


class Hinge(LossFunction):
    """Loss function for trainning binary linear classifiers with target
    {-1, 1}. It computes the aggregated loss, not the average per sample.
    """

    def __init__(self, threshold=1.0):
        self.threshold = threshold
        super(Hinge, self).__init__()

    def loss(self, y_pred, y_true):
        z = y_pred * y_true
        losses = (self.threshold - z)*(z <= self.threshold).astype(float)
        loss = np.mean(losses)
        return loss

    def gradient_loss(self, y_pred, y, x):
        """Derivation dL/dw. It is separated the output for w and w0.
        """
        z = y_pred * y
        in_margin = (z <= self.threshold).astype(float)

        gradloss_w = -np.dot(y*in_margin, x)
        gradloss_w0 = -np.sum(y*in_margin)

        return gradloss_w, gradloss_w0


################################ Regularization ###############################
###############################################################################
class Regularization(object):

    @classmethod
    def create_regularization(cls, regularizer, *parameters):
        regularizations =\
            {'null': Null_Regularization, 'l2': L2_Regularization}
        if type(regularizer) == str:
            return regularizations[regularizer.lower()](*parameters)
        if isinstance(regularizer, Regularization):
            return regularizer
        else:
            return regularizer(*parameters)


class Null_Regularization(Regularization):

    def __init__(self):
        pass

    @property
    def parameters(self):
        return None

    def regularize(self, *args):
        return 0.

    def gradient_regularization(self, *args):
        return 0.


class L1_Regularization(Regularization):

    def __init__(self, lambda_par):
        self.lambda_par = lambda_par

    @property
    def parameters(self):
        return self.lambda_par

    def regularize(self, w, w0=0.):
        return 0.5*(np.sqrt(np.dot(w, w))+w0)*self.lambda_par

    def gradient_regularization(self, w, w0=0.):
        return self.lambda_par, self.lambda_par


class L2_Regularization(Regularization):

    def __init__(self, lambda_par):
        self.lambda_par = lambda_par

    @property
    def parameters(self):
        return self.lambda_par

    def regularize(self, w, w0=0.):
        return 0.5*(np.dot(w, w)+w0**2)*self.lambda_par

    def gradient_regularization(self, w, w0=0.):
        return self.lambda_par*w, self.lambda_par*w0
