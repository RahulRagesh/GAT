from inits import *
import tensorflow as tf

def dropout(x, keep_prob, seed=123):
    if isinstance(x, tf.SparseTensor):
        values = x.values 
        values = tf.nn.dropout(values, keep_prob, seed=seed)
        res = tf.SparseTensor(x.indices, values, x.dense_shape)
    else:
        res = tf.nn.dropout(x, keep_prob, seed=seed)
    return res


def dot(x, y):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if isinstance(x, tf.SparseTensor):
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
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(kwargs['parent_model'].get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

class GraphAttention(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, model_dropout, attention_dropout,
                 act=tf.nn.relu, bias=False, attention_bias=False, **kwargs):
        super(GraphAttention, self).__init__(**kwargs)

        self.seed = kwargs['parent_model'].seed
        self.act = act

        self.adj = placeholders['adj']
       
        self.model_dropout = placeholders['model_dropout'] if model_dropout else 0.
        self.bias = bias

        self.attention_dropout = placeholders['attention_dropout'] if attention_dropout else 0.
        self.attention_bias = attention_bias

        self.num_heads = kwargs.get('num_heads',1)
        self.average_heads = kwargs.get('average_heads',False)

        with tf.variable_scope(self.name + '_vars'):
            for i in range(self.num_heads):
                self.vars['encoder_weights_' + str(i)] = glorot([input_dim, output_dim], name='encoder_weights_' + str(i), seed=self.seed)
                tf.add_to_collection('MODEL_WEIGHTS', self.vars['encoder_weights_' + str(i)])


                self.vars['attention_weights_'+str(i)] = glorot([2*output_dim, 1], name='attention_weights_' + str(i), seed=self.seed)
                tf.add_to_collection('ATTENTION_WEIGHTS', self.vars['attention_weights_' + str(i)])

                if self.attention_bias:
                    self.vars['attention_bias_'+str(i)] = zeros([1], name='attention_bias_'+str(i))
                    tf.add_to_collection('ATTENTION_WEIGHTS', self.vars['attention_bias_' + str(i)])


            if self.bias:
                self.vars['model_bias'] = zeros([output_dim*self.num_heads], name='model_bias')
                tf.add_to_collection('MODEL_WEIGHTS', self.vars['model_bias'])

    def get_attention_coefficients(self, features, head):
        indices = self.adj.indices 
        pairwise_features = tf.concat([tf.gather(features,indices[:,0]),tf.gather(features,indices[:,1])],axis=1)
        attention_coefficients = tf.matmul(pairwise_features, self.vars['attention_weights_'+str(head)])
        if self.attention_bias:
            attention_coefficients = attention_coefficients + self.vars['attention_bias_'+str(head)]
        attention_coefficients = tf.nn.leaky_relu(attention_coefficients, alpha=0.2)
        attention_coefficients = tf.reshape(attention_coefficients, (-1,))
        attention_matrix = tf.SparseTensor(indices=indices,values=attention_coefficients,dense_shape=self.adj.dense_shape)
        attention_matrix = tf.sparse_reorder(attention_matrix)
        attention_matrix = tf.sparse_softmax(attention_matrix)
        return  attention_matrix

    def _call(self, inputs):
        x = inputs
        x = dropout(x, 1-self.model_dropout)

        self.attention_matrices = list()
        self.head_outputs = list()
        for i in range(self.num_heads):
            transformed_features = dot(x, self.vars['encoder_weights_' + str(i)])
            
            attention_matrix = self.get_attention_coefficients(transformed_features, i)
            attention_matrix = dropout(attention_matrix, 1-self.attention_dropout)
            self.attention_matrices.append(attention_matrix)
            head_output = dot(self.attention_matrices[-1], transformed_features)
            
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
