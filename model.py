import tensorflow as tf

from tensorflow.python.layers.core import Dropout, Dense


class LSTM_Model():
    def __init__(self, input_shape, lr, attn_fusion=True, unimodal=False, seed=1227):
        if unimodal:
            self.input = tf.placeholder(dtype=tf.float32, shape=(None, input_shape[0], input_shape[1]))
        else:
            self.a_input = tf.placeholder(dtype=tf.float32, shape=(None, input_shape[0], input_shape[1]))
            self.v_input = tf.placeholder(dtype=tf.float32, shape=(None, input_shape[0], input_shape[1]))
            self.t_input = tf.placeholder(dtype=tf.float32, shape=(None, input_shape[0], input_shape[1]))
        self.mask = tf.placeholder(dtype=tf.float32, shape=(None, input_shape[0]))
        self.seq_len = tf.placeholder(tf.int32, [None, ], name="seq_len")
        self.y = tf.placeholder(tf.int32, [None, input_shape[0], 2], name="y")
        self.lr = lr
        self.seed = seed
        self.attn_fusion = attn_fusion
        self.unimodal = unimodal
        self.lstm_dropout = tf.placeholder(tf.float32, name="lstm_dropout")
        self.dropout = tf.placeholder(tf.float32, name="dropout")

        # Build the model
        self._build_model_op()
        self._initialize_optimizer()

    def BiGRU(self, inputs, output_size, name, dropout_keep_rate):
        with tf.variable_scope('rnn_' + name, reuse=tf.AUTO_REUSE):
            kernel_init = tf.glorot_uniform_initializer(seed=self.seed, dtype=tf.float32)
            bias_init = tf.zeros_initializer()

            fw_cell = tf.contrib.rnn.GRUCell(output_size, name='gru', reuse=tf.AUTO_REUSE, activation=tf.nn.tanh,
                                             kernel_initializer=kernel_init, bias_initializer=bias_init)
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_rate)

            bw_cell = tf.contrib.rnn.GRUCell(output_size, name='gru', reuse=tf.AUTO_REUSE, activation=tf.nn.tanh,
                                             kernel_initializer=kernel_init, bias_initializer=bias_init)
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_rate)

            output_fw, _ = tf.nn.dynamic_rnn(fw_cell, inputs, sequence_length=self.seq_len, dtype=tf.float32)
            output_bw, _ = tf.nn.dynamic_rnn(bw_cell, inputs, sequence_length=self.seq_len, dtype=tf.float32)

            output = tf.concat([output_fw, output_bw], axis=-1)
            return output

    def self_attention(self, inputs_a, inputs_v, inputs_t, name):
        """

        :param inputs_a: audio input (B, T, dim)
        :param inputs_v: video input (B, T, dim)
        :param inputs_t: text input (B, T, dim)
        :param name: scope name
        :return:
        """
        inputs_a = tf.expand_dims(inputs_a, axis=1)
        inputs_v = tf.expand_dims(inputs_v, axis=1)
        inputs_t = tf.expand_dims(inputs_t, axis=1)
        # inputs = (B, 3, T, dim)
        inputs = tf.concat([inputs_a, inputs_v, inputs_t], axis=1)
        t = inputs.get_shape()[2].value
        share_param = True
        hidden_size = inputs.shape[-1].value  # D value - hidden size of the RNN layer
        if share_param:
            scope_name = 'self_attn'
        else:
            scope_name = 'self_attn' + name
        # print(scope_name)
        inputs = tf.transpose(inputs, [2, 0, 1, 3])
        with tf.variable_scope(scope_name):
            outputs = []
            for x in range(t):
                t_x = inputs[x, :, :, :]
                # t_x => B, 3, dim
                den = True
                if den:
                    x_proj = Dense(hidden_size)(t_x)
                    x_proj = tf.nn.tanh(x_proj)
                else:
                    x_proj = t_x
                u_w = tf.Variable(tf.random_normal([hidden_size, 1], stddev=0.01, seed=1227))
                x = tf.tensordot(x_proj, u_w, axes=1)
                alphas = tf.nn.softmax(x, axis=-1)
                output = tf.matmul(tf.transpose(t_x, [0, 2, 1]), alphas)
                output = tf.squeeze(output, -1)
                outputs.append(output)

            final_output = tf.stack(outputs, axis=1)
            # print('final_output', final_output.get_shape())
            return final_output

    def _build_model_op(self):
        # self attention
        if self.unimodal:
            input = self.input
        else:
            if self.attn_fusion:
                input = self.self_attention(self.a_input, self.v_input, self.t_input, '')
            else:
                input = tf.concat([self.a_input, self.v_input, self.t_input], axis=-1)

        gru_output = self.BiGRU(input, 300, 'gru', 1 - self.lstm_dropout)
        inter = Dropout(self.dropout)(gru_output)
        init = tf.glorot_uniform_initializer(seed=self.seed, dtype=tf.float32)
        if self.unimodal:
            self.inter1 = Dense(100, activation=tf.nn.tanh,
                                kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed, dtype=tf.float32))(
                inter)
        else:
            self.inter1 = Dense(500, activation=tf.nn.relu,
                                kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed, dtype=tf.float32))(
                inter)
        inter = Dropout(self.dropout)(self.inter1)
        self.output = Dense(2, kernel_initializer=init)(inter)
        # print('self.output', self.output.get_shape())
        self.preds = tf.nn.softmax(self.output)

        # To calculate the number correct, we want to count padded steps as incorrect
        correct = tf.cast(
            tf.equal(tf.argmax(self.preds, -1, output_type=tf.int32), tf.argmax(self.y, -1, output_type=tf.int32)),
            tf.int32) * tf.cast(self.mask, tf.int32)

        # To calculate accuracy we want to divide by the number of non-padded time-steps,
        # rather than taking the mean
        self.accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.reduce_sum(tf.cast(self.seq_len, tf.float32))
        y = tf.argmax(self.y, -1)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=y)
        loss = loss * self.mask

        self.loss = tf.reduce_sum(loss) / tf.reduce_sum(self.mask)

    def _initialize_optimizer(self):
        self.global_step = tf.get_variable(shape=[], initializer=tf.constant_initializer(0), dtype=tf.int32,
                                           name='global_step')
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999)
        # self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=None)

        self.train_op = self._optimizer.minimize(self.loss, global_step=self.global_step)
