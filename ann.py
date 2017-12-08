import tensorflow as tf

class ANN(object):
    """A Fully-Connected ANN to classify bee hive audio"""
    def __init__(self, N=256, num_hidden=3, hidden_nodes=256, num_classes=3):
        self.N = N
        self.num_hidden = num_hidden
        self.hidden_nodes = hidden_nodes
        self.num_classes = num_classes
        self.x = tf.placeholder(tf.float32, shape=[None, N], name='input_audio')
        self.y_ = tf.placeholder(tf.float32, shape=[None, 3], name='audio_class')

    def _weight_variables(self, shape, name=''):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def _bias_variable(self, shape, name=''):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def _layer(self, x, weights, biases):
        return tf.add(tf.matmul(x, weights), biases)

    def build(self):
        N = self.N
        hiddenN = self.hidden_nodes
        with tf.variable_scope("model", initializer=tf.random_uniform_initializer()) as scope:
            # Build the input layer
            last_layer = self._layer(self.x, self._weight_variables([N,hiddenN], 'input_layer'), self._bias_variable([hiddenN], 'input_bias'))

            # Build the hidden layers
            for i in range(self.num_hidden):
                last_layer = self._layer(last_layer, self._weight_variables([hiddenN,hiddenN], 'hidden_layer_' + str(i)), self._bias_variable([hiddenN], 'hidden_bias_' + str(i)))
        
            # Build the logits of the ANN
            self.logits = self._layer(last_layer, self._weight_variables([hiddenN,self.num_classes], 'output_layer'), self._bias_variable([self.num_classes], 'output_bias'))
            self.prediction = tf.nn.softmax(self.logits)

        with tf.variable_scope("training", initializer=tf.random_uniform_initializer()) as scope:
            # Build Training stuff
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_))
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

            self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.saver = tf.train.Saver()

