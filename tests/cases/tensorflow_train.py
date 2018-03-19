import numpy as np
import glob
import os
from gunpowder import *
from gunpowder.tensorflow import Train, Predict
# from gunpowder.ext import tensorflow as tf
import tensorflow as tf
from .provider_test import ProviderTest

class TestTensorflowTrainSource(BatchProvider):

    def setup(self):

        spec = ArraySpec(
            roi=Roi((0, 0), (2, 2)),
            dtype=np.float32,
            interpolatable=True,
            voxel_size=(1, 1))
        self.provides(ArrayKeys.A, spec)
        self.provides(ArrayKeys.B, spec)

    def provide(self, request):

        batch = Batch()

        spec = self.spec[ArrayKeys.A]
        spec.roi = request[ArrayKeys.A].roi

        batch.arrays[ArrayKeys.A] = Array(
            np.array([[0, 1], [2, 3]], dtype=np.float32),
            spec)

        spec = self.spec[ArrayKeys.B]
        spec.roi = request[ArrayKeys.B].roi

        batch.arrays[ArrayKeys.B] = Array(
            np.array([[0, 1], [2, 3]], dtype=np.float32),
            spec)

        return batch

class TestTensorflowTrain(ProviderTest):

    def create_meta_graph(self):

        # create a tf graph
        a = tf.placeholder(tf.float32, shape=(2, 2))
        b = tf.placeholder(tf.float32, shape=(2, 2))
        v = tf.Variable(1, dtype=tf.float32)
        c = a*b*v

        # dummy "loss"
        loss = tf.norm(c)

        # dummy optimizer
        opt = tf.train.AdamOptimizer()
        optimizer = opt.minimize(loss)

        tf.train.export_meta_graph(filename='tf_graph.meta')

        return [x.name for x in [a, b, c, optimizer, loss]]

    def test_output(self):

        # start clean
        for filename in glob.glob('tf_graph.*'):
            os.remove(filename)
        for filename in glob.glob('tf_graph_checkpoint_*'):
            os.remove(filename)
        try:
            os.remove('checkpoint')
        except:
            pass

        ArrayKey('A')
        ArrayKey('B')
        ArrayKey('C')
        ArrayKey('GRADIENT_A')

        # create model meta graph file and get input/output names
        (a, b, c, optimizer, loss) = self.create_meta_graph()

        source = TestTensorflowTrainSource()
        train = Train(
            'tf_graph',
            optimizer=optimizer,
            loss=loss,
            inputs={a: ArrayKeys.A, b: ArrayKeys.B},
            outputs={c: ArrayKeys.C},
            gradients={a: ArrayKeys.GRADIENT_A},
            save_every=100)
        pipeline = source + train

        request = BatchRequest({
            ArrayKeys.A: ArraySpec(roi=Roi((0, 0), (2, 2))),
            ArrayKeys.B: ArraySpec(roi=Roi((0, 0), (2, 2))),
            ArrayKeys.C: ArraySpec(roi=Roi((0, 0), (2, 2))),
            ArrayKeys.GRADIENT_A: ArraySpec(roi=Roi((0, 0), (2, 2))),
        })

        # train for a couple of iterations
        with build(pipeline):

            batch = pipeline.request_batch(request)

            self.assertAlmostEqual(batch.loss, 9.8994951)

            gradient_a = batch.arrays[ArrayKeys.GRADIENT_A].data
            self.assertTrue(gradient_a[0, 0] < gradient_a[0, 1])
            self.assertTrue(gradient_a[0, 1] < gradient_a[1, 0])
            self.assertTrue(gradient_a[1, 0] < gradient_a[1, 1])

            for i in range(200-1):
                loss1 = batch.loss
                batch = pipeline.request_batch(request)
                loss2 = batch.loss
                self.assertLess(loss2, loss1)

        # resume training
        with build(pipeline):

            for i in range(100):
                loss1 = batch.loss
                batch = pipeline.request_batch(request)
                loss2 = batch.loss
                self.assertLess(loss2, loss1)

        # predict
        source = TestTensorflowTrainSource()
        predict = Predict(
            'tf_graph_checkpoint_300',
            inputs={a: ArrayKeys.A, b: ArrayKeys.B},
            outputs={c: ArrayKeys.C})
        pipeline = source + predict

        request = BatchRequest({
            ArrayKeys.A: ArraySpec(roi=Roi((0, 0), (2, 2))),
            ArrayKeys.B: ArraySpec(roi=Roi((0, 0), (2, 2))),
            ArrayKeys.C: ArraySpec(roi=Roi((0, 0), (2, 2))),
        })

        with build(pipeline):

            prev_c = None

            for i in range(100):
                batch = pipeline.request_batch(request)
                c = batch.arrays[ArrayKeys.C].data

                if prev_c is not None:
                    self.assertTrue(np.equal(c, prev_c))
                    prev_c = c
