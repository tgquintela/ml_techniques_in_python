import numpy as np
import unittest
from itertools import product

from ml_techniques.svm import *


class PermutationDataTest(unittest.TestCase):

    def testpropershape(self):
        data = np.random.random((10, 4))
        labels = np.random.randint(0, 2, 10)*2-1

        data_per = permut_data(data)
        self.assertEqual(data_per.shape, data.shape)

        data_per, labels_per = permut_data(data, labels)
        self.assertEqual(data_per.shape, data.shape)
        self.assertEqual(labels_per.shape, labels.shape)


class BatchCreatorTest(unittest.TestCase):

    def test_run_batch_iterator(self):
        data_size = 100
        batch_size = 9
        for init, endit in batch_size_iter(data_size, batch_size):
            self.assertTrue(init != endit)
            self.assertTrue(init < endit)
        self.assertEqual(endit, data_size)

        data_size = 100
        batch_size = 10
        for init, endit in batch_size_iter(data_size, batch_size):
            self.assertTrue(init != endit)
            self.assertTrue(init < endit)
        self.assertEqual(endit, data_size)


class RegularizationTest(unittest.TestCase):

    def assert_regularization(self, reg):
        reg.parameters
        reg.regularize(np.random.randn(10), 1)
        reg.gradient_regularization(np.random.randn(10), 1)

    def test_abstractregularization(self):
        reg = Regularization.create_regularization('l2', 1.)
        self.assert_regularization(reg)
        reg = Regularization.create_regularization(reg)
        self.assert_regularization(reg)
        reg = Regularization.create_regularization(Null_Regularization)
        self.assert_regularization(reg)

    def test_l2_regularization(self):
        reg = L1_Regularization(1.)
        self.assert_regularization(reg)

    def test_l1_regularization(self):
        reg = L1_Regularization(1.)
        self.assert_regularization(reg)


class AccuracyFunctionTest(unittest.TestCase):

    def test_order_independency(self):
        n = 10
        n_tests = 20

        for i in range(n_tests):
            y0 = np.random.randint(0, 2, n)
            y1 = np.random.randint(0, 2, n)
            reindices = np.random.permutation(n)
            self.assertEqual(accuracy(y0, y1),
                             accuracy(y0[reindices], y1[reindices]))

    def test_symetry(self):
        n = 10
        n_tests = 20

        for i in range(n_tests):
            y0 = np.random.randint(0, 2, n)
            y1 = np.random.randint(0, 2, n)
            self.assertEqual(accuracy(y0, y1), accuracy(y1, y0))


class LossFunctionTest(unittest.TestCase):

    def _generator_labels(self, n):
        return np.random.randint(0, 2, n)*2-1

    def test_abstractloss(self):
        lossf = LossFunction.create_lossfunction('Hinge')
        lossf = LossFunction.create_lossfunction(lossf)
        lossf = LossFunction.create_lossfunction(Hinge)

    def test_loss(self):
        n = 20
        y0 = np.random.random(n)*2-1
        y1 = self._generator_labels(n)

        thresholds = [0, 1, 2]
        for thr in thresholds:
            lossf = Hinge(thr)
            lossf.loss(y0, y1)

    def test_gradient(self):
        n, n_feats = 20, 10
        y0 = np.random.random(n)*2-1
        y1 = self._generator_labels(n)
        x = np.random.random((n, n_feats))

        thresholds = [0, 1, 2]
        for thr in thresholds:
            lossf = Hinge(thr)
            grad_w, grad_w0 = lossf.gradient_loss(y0, y1, x)
            self.assertEqual(len(grad_w), n_feats)


class Modeltest(unittest.TestCase):

    def setUp(self):
        n = 100
        self.create_X = lambda n_feats: np.random.random((n, n_feats))

    def assert_linearmodel(self, linearmodel):
        w, w0 = linearmodel.parameters
        if w is not None:
            linearmodel.compute(self.create_X(len(w)))
        linearmodel.reset_model()

    def test_abstractmodel(self):
        mod = Model.create_model('svm', np.random.randn(10), 0.)
        Model.create_model(mod)
        Model.create_model(LinearModel, np.random.randn(10), 0.)

    def test_linearmodel(self):
        lm = LinearModel(None)
        self.assert_linearmodel(lm)
        lm = LinearModel(np.random.randn(10), 0.)
        self.assert_linearmodel(lm)
        lm = LinearModel.weights_initialization(10, 'gauss')
        self.assert_linearmodel(lm)
        lm = LinearModel.weights_initialization(10, 'zeros')
        self.assert_linearmodel(lm)


class SVMTest(unittest.TestCase):

    def setUp(self):
        loss = ['Hinge', Hinge()]
        reg_pars = [0.01, 1., 10.]
        batch_size = [10]
        n_epochs = [0, 100]
        learning_rate = [0.001, 1.]
        stop_step = [.00001, 100]
        history = [True, False]
        verbose = [True, False]

        self.var_names = ['loss', 'reg_pars', 'batch_size', 'n_epochs',
                          'learning_rate', 'stop_step', 'history', 'verbose']
        self.possibilities = [loss, reg_pars, batch_size, n_epochs,
                              learning_rate, stop_step, history, verbose]

    def test_initialization(self):
        n, n_feats = 100, 20
        data = np.random.random((n, n_feats))
        labels = np.random.randint(0, 2, n)*2-1

        for p in product(*self.possibilities):
            solver = SVM(**dict(zip(self.var_names, p)))
            ## General asserts
            self.assertEqual(solver.optimizer, 'SGD')
            self.assertEqual(solver.batch_size, p[2])
            self.assertEqual(solver.n_epochs, p[3])
            self.assertEqual(solver.learning_rate, p[4])
            self.assertEqual(solver.stop_step, p[5])

            ## Special cases
            if not p[6]:
                self.assertIsNone(solver.train_loss_history)
                self.assertIsNone(solver.test_loss_history)
                self.assertIsNone(solver.train_accuracy_history)
                self.assertIsNone(solver.test_accuracy_history)

            ## Weights initialization
            solver.model = solver.model.weights_initialization(n_feats)
            solver._reset_history()

            ## Batch creation testing
            for x_batch, y_batch in solver._batch_generator(data, labels):
                self.assertTrue(len(x_batch) >= p[2])

            ## Computer functions
            if p[7]:
#                model._initialization_weights(n_feats, init_type='gauss')
                solver.compute_epoch_measures(data, labels, None, None)
                solver.compute_epoch_measures(data, labels, data, labels)

    def test_fitmodel(self):
        n, n_feats = 100, 5
        data = np.random.random((n, n_feats))
        labels = np.random.randint(0, 2, n)*2-1

        for p in product(*self.possibilities):
            solver = SVM(**dict(zip(self.var_names, p)))

            solver.report_results()
            solver.n_epochs = 100
            solver.fit(data, labels)
            solver.fit(data, labels, data, labels)
            solver.predict(data)
            solver.score(data, labels)
            if p[6]:
                self.assertEqual(solver.epoch_learned,
                                 len(solver.train_loss_history))
                self.assertEqual(solver.epoch_learned,
                                 len(solver.train_accuracy_history))
                self.assertEqual(solver.epoch_learned,
                                 len(solver.test_loss_history))
                self.assertEqual(solver.epoch_learned,
                                 len(solver.test_accuracy_history))
            solver.report_results()
