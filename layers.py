from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging','num_heads','average_heads'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class GraphAttention(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, model_dropout, attention_dropout,
                 sparse_inputs=False, act=tf.nn.relu, bias=False, attention_bias=False,
                 featureless=False, **kwargs):
        super(GraphAttention, self).__init__(**kwargs)

        self.adj = placeholders['adj']
        
        self.model_dropout = placeholders['model_dropout'] if model_dropout else 0.
        self.bias = bias

        self.attention_dropout = placeholders['attention_dropout'] if attention_dropout else 0.
        self.attention_bias = attention_bias
        self.num_heads = kwargs.get('num_heads',1)

        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.act = act

        self.average_heads = kwargs.get('average_heads',False)

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(self.num_heads):
                self.vars['encoder_weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='encoder_weights_' + str(i))
                tf.add_to_collection('MODEL_WEIGHTS', self.vars['encoder_weights_' + str(i)])


                self.vars['attention_weights_'+str(i)] = glorot([2*output_dim, 1],
                                                        name='attention_weights_' + str(i))
                tf.add_to_collection('ATTENTION_WEIGHTS', self.vars['attention_weights_' + str(i)])

                if self.attention_bias:
                    self.vars['attention_bias_'+str(i)] = zeros([1], name='attention_bias_'+str(i))
                    tf.add_to_collection('ATTENTION_WEIGHTS', self.vars['attention_bias_' + str(i)])


            if self.bias:
                self.vars['model_bias'] = zeros([output_dim*self.num_heads], name='model_bias')
                tf.add_to_collection('MODEL_WEIGHTS', self.vars['model_bias'])

        if self.logging:
            self._log_vars()

    def get_attention_coefficients(self, features, head):
        indices = self.adj.indices 
        pairwise_features = tf.concat([tf.gather(features,indices[:,0]),tf.gather(features,indices[:,1])],axis=1)
        attention_coefficients = tf.matmul(pairwise_features, self.vars['attention_weights_'+str(head)])
       
        if self.attention_bias:
            attention_coefficients = attention_coefficients + self.vars['attention_bias_'+str(head)]
       
        attention_coefficients = tf.nn.leaky_relu(attention_coefficients, alpha=0.2)
        attention_coefficients = tf.reshape(attention_coefficients, (-1,))
        attention_coefficients = tf.nn.dropout(attention_coefficients, 1-self.attention_dropout)
        
        attention_matrix = tf.SparseTensor(indices=indices,values=attention_coefficients,dense_shape=self.adj.dense_shape)
        attention_matrix = tf.sparse_softmax(attention_matrix)
        
        return  attention_matrix

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.model_dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.model_dropout)

        self.attention_matrices = list()
        self.head_outputs = list()
        for i in range(self.num_heads):
            transformed_features = dot(x, self.vars['encoder_weights_' + str(i)], sparse=self.sparse_inputs)
            
            attention_matrix = self.get_attention_coefficients(transformed_features, i)
            self.attention_matrices.append(attention_matrix)
            
            head_output = dot(attention_matrix, transformed_features, sparse=True)
            if not self.average_heads:
                head_output = self.act(head_output)

            self.head_outputs.append(head_output)
        
        if self.average_heads:
            output = self.act(tf.add_n(self.head_outputs)/self.num_heads)
        else:
            output = tf.concat(self.head_outputs,axis=1)
        
        # bias
        if self.bias:
            output += self.vars['model_bias']

        return output
