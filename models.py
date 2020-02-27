from layers import *
from metrics import *

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging','num_heads'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging
        self._LAYER_UIDS = {}

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def get_layer_uid(self,layer_name=''):
        """Helper function, assigns unique layer IDs."""
        if layer_name not in self._LAYER_UIDS:
            self._LAYER_UIDS[layer_name] = 1
            return 1
        else:
            self._LAYER_UIDS[layer_name] += 1
            return self._LAYER_UIDS[layer_name]

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

class GAT(Model):
    def __init__(self, configs, placeholders, input_dim, **kwargs):
        super(GAT, self).__init__(**kwargs)
        self.configs = configs
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=configs['learning_rate'])
        self.num_heads = configs['num_heads']
        self.build()

    def _loss(self):
        # Weight decay loss
        for layer in self.layers:
            for var in layer.vars.values():
                if 'weight' in var.name:
                    self.loss += self.configs['weight_decay'] * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphAttention(input_dim=self.input_dim,
                                            output_dim=self.configs['hidden_dims'],
                                            num_heads = self.num_heads,
                                            average_heads=False,
                                            sparse_inputs=True,
                                            act=tf.nn.elu,
                                            placeholders=self.placeholders,
                                            model_dropout=True,
                                            attention_dropout=True,
                                            bias=True,
                                            attention_bias=True,
                                            logging=self.logging,
                                            parent_model=self))

        self.layers.append(GraphAttention(input_dim=self.configs['hidden_dims']*self.num_heads,
                                            output_dim=self.output_dim,
                                            num_heads=1,
                                            average_heads=True,
                                            sparse_inputs=False,
                                            act=lambda x: x,
                                            placeholders=self.placeholders,
                                            model_dropout=True,
                                            attention_dropout=True,
                                            bias=True,
                                            attention_bias=True,
                                            logging=self.logging,
                                            parent_model=self))

    def predict(self):
        return tf.nn.softmax(self.outputs)
