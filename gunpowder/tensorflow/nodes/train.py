import logging
import os
import numpy as np

from gunpowder.ext import tensorflow as tf
from gunpowder.nodes.generic_train import GenericTrain
from gunpowder.array import ArrayKey, Array


import code

logger = logging.getLogger(__name__)

class Train(GenericTrain):
    '''Tensorflow implementation of :class:`gunpowder.nodes.Train`.

    Args:

        graph: Filename of a tensorflow meta-graph storing the
            tensorflow graph containing an optimizer. A meta-graph file can be
            created by running::

                # create tensorflow graph
                ...

                # store it
                tf.train.export_meta_graph(filename='...')

        optimizer (string or function): Either the name of the tensorflow
            operator performing a training iteration, or a function that, given
            the graph of the meta-graph file, adds a custom loss and optimizer.

            If a function is given, it should return a tuple ``(loss,
            optimizer)`` of a tensor and an operator representing the loss and
            the optimizer, respectively. In this case, parameter ``loss``
            should be ``None``.

            Example::

                def add_custom_optimizer(graph):

                    # get the output of your graph
                    output = graph.get_tensor_by_name('...')

                    # create your custom loss
                    loss = custom_loss(output)

                    # add an optimizer of your choice
                    optimizer = tf.train.AdamOptimizer().minimize(loss)

                    return (loss, optimizer)

        loss (string or None): The name of the tensorflow tensor containing the
            loss, or ``None`` if ``optimizer`` is a function.

        inputs (dict): Dictionary from the names of input tensors in the
            network to :class:``ArrayKey`` or batch attribute name as string.

        outputs (dict): Dictionary from the names of output tensors in the
            network to :class:``ArrayKey``. New arrays will be generated by
            this node for each entry (if requested downstream).

        gradients (dict): Dictionary from the names of output tensors in the
            network to :class:``ArrayKey``. New arrays containing the
            gradient of an output with respect to the loss will be generated by
            this node for each entry (if requested downstream).

        summary (string or None): The name of the tensorflow tensor containing
            the tensorboard summaries. ``None`` if not using tensorboard.

        array_specs (dict, optional): An optional dictionary of
            :class:`ArrayKey` to :class:`ArraySpec` to set the array specs
            of generated arrays (``outputs`` and ``gradients``). This is useful
            to set the ``voxel_size``, for example, if they differ from the
            voxel size of the input arrays. Only fields that are not ``None``
            in the given :class:`ArraySpec` will be used.

        save_every (int, optional): After how many iterations to create a
            checkpoint to store the learnt weights.

        log_dir (string, optional): Directory for saving tensorboard
            summaries.
    '''

    def __init__(
            self,
            graph,
            optimizer,
            loss,
            inputs,
            outputs,
            gradients,
            summary=None,
            array_specs=None,
            save_every=2000,
            log_dir='./',
            checkpoint_dir='./'
            ):

        super(Train, self).__init__(
            inputs,
            outputs,
            gradients,
            array_specs,
            spawn_subprocess=False)
        self.meta_graph_filename = graph
        self.optimizer_func = None
        self.optimizer_loss_names = None
        self.optimizer = None
        self.loss = None
        self.summary = summary
        self.session = None
        self.tf_gradient = {}
        self.graph = None
        self.basic_saver = None
        self.full_saver = None
        self.save_every = save_every
        self.iteration = None
        self.iteration_increment = None
        self.summary_saver = None
        self.log_dir = log_dir
        self.checkpoint_dir= checkpoint_dir
        if isinstance(optimizer, basestring):
            self.optimizer_loss_names = (optimizer, loss)
        else:
            self.optimizer_func = optimizer

    def start(self):

        logger.info("Initializing tf session...")

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        with self.graph.as_default():
            self.__read_meta_graph()

        if self.summary is not None:
            self.summary_saver = tf.summary.FileWriter(self.log_dir, self.graph)

        if self.optimizer_func is None:

            # get actual operations/tensors from names
            self.optimizer = self.graph.get_operation_by_name(self.optimizer_loss_names[0])
            self.loss = self.graph.get_tensor_by_name(self.optimizer_loss_names[1])

        # add symbolic gradients
        for tensor_name in self.gradients:
            tensor = self.graph.get_tensor_by_name(tensor_name)
            self.tf_gradient[tensor_name] = tf.gradients(
                self.loss,
                [tensor])[0]

    def train_step(self, batch, request):

        array_outputs = self.__collect_requested_outputs(request)
        inputs = self.__collect_provided_inputs(batch)

        to_compute = {
            'optimizer': self.optimizer,
            'loss': self.loss,
            'iteration': self.iteration_increment}
        to_compute.update(array_outputs)


        # compute outputs, gradients, and update variables
        if self.summary is not None:
            outputs, summaries = self.session.run([to_compute, self.summary], feed_dict=inputs)
        else:
            outputs = self.session.run(to_compute, feed_dict=inputs)

        for array_key in array_outputs:
            spec = self.spec[array_key].copy()
            spec.roi = request[array_key].roi
            batch.arrays[array_key] = Array(
                outputs[array_key],
                spec)

        batch.loss = outputs['loss']
        batch.iteration = outputs['iteration'][0]


        if self.summary is not None:
            self.summary_saver.add_summary(summaries, batch.iteration)

        if batch.iteration%self.save_every == 0:

            checkpoint_name = (self.checkpoint_dir+
                self.meta_graph_filename +
                '_checkpoint_%i'%batch.iteration)

            logger.info(
                "Creating checkpoint %s",
                checkpoint_name)


            self.full_saver.save(
                self.session,
                checkpoint_name)

    def stop(self):

        if self.session is not None:

            self.optimizer = None
            self.loss = None
            if self.summary is not None:
                self.summary_saver.close()
            self.session.close()
            self.graph = None
            self.session = None

    def __read_meta_graph(self):

        logger.info("Reading meta-graph...")

        # read the original meta-graph
        tf.train.import_meta_graph(
            self.meta_graph_filename + '.meta',
            clear_devices=True)

        # add custom gunpowder variables
        with tf.variable_scope('gunpowder'):
            self.iteration = tf.get_variable(
                'iteration',
                shape=1,
                initializer=tf.zeros_initializer,
                trainable=False)
            self.iteration_increment = tf.assign(
                self.iteration,
                self.iteration + 1)

        # Until now, only variables have been added to the graph that are part
        # of every checkpoint. We create a 'basic_saver' for only those
        # variables.
        self.basic_saver = tf.train.Saver(max_to_keep=None)

        # Add custom optimizer and loss, if requested. This potentially adds
        # more variables, not covered by the basic_saver.
        if self.optimizer_func is not None:
            loss, optimizer = self.optimizer_func(self.graph)
            self.loss = loss
            self.optimizer = optimizer

        # We create a 'full_saver' including those variables.
        self.full_saver = tf.train.Saver(max_to_keep=None)

        # find most recent checkpoint
        checkpoint_dir = os.path.dirname(self.meta_graph_filename)
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

        if checkpoint:

            try:
                # Try to restore the graph, including the custom optimizer
                # state (if a custom optimizer was used).
                self.__restore_graph(checkpoint, restore_full=True)

            except tf.errors.NotFoundError:

                # If that failed, we just transitioned from an earlier training
                # without the custom optimizer. In this case, restore only the
                # variables of the original meta-graph and 'gunpowder'
                # variables. Custom optimizer variables will be default
                # initialized.
                logger.info("Checkpoint did not contain custom optimizer "
                            "variables")
                self.__restore_graph(checkpoint, restore_full=False)
        else:

            logger.info("No checkpoint found")

            # initialize all variables
            self.session.run(tf.global_variables_initializer())

    def __restore_graph(self, checkpoint, restore_full):

        logger.info("Restoring model from %s", checkpoint)

        if restore_full:

            logger.info("...using a saver for all variables")
            self.full_saver.restore(self.session, checkpoint)

        else:

            # initialize all variables, such that non-basic variables are
            # initialized
            self.session.run(tf.global_variables_initializer())

            logger.info("...using a saver for basic variables only")
            self.basic_saver.restore(self.session, checkpoint)

    def __collect_requested_outputs(self, request):

        array_outputs = {}

        for output_name, array_key in self.outputs.items():
            if array_key in request:
                array_outputs[array_key] = output_name

        for output_name, array_key in self.gradients.items():
            if array_key in request:
                array_outputs[array_key] = self.tf_gradient[output_name]


        return array_outputs

    def __collect_provided_inputs(self, batch):

        inputs = {}

        for input_name, input_key in self.inputs.items():
            if isinstance(input_key, ArrayKey):
                if input_key in batch.arrays:
                    inputs[input_name] = batch.arrays[input_key].data
                else:
                    logger.warn("batch does not contain %s, input %s will not "
                                "be set", input_key, input_name)
            elif isinstance(input_key, np.ndarray):
                inputs[input_name] = input_key
            elif isinstance(input_key, str):
                inputs[input_name] = getattr(batch, input_key)
            else:
                raise Exception(
                    "Unknown network input key {}, can't be given to "
                    "network".format(input_key))

        return inputs
