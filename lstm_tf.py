import argparse
import pickle
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dropout, Dense
from tqdm import tqdm

unimodal_activations = {}


def createOneHot(train_label, test_label):
    maxlen = int(max(train_label.max(), test_label.max()))

    train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            train[i, j, train_label[i, j]] = 1

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            test[i, j, test_label[i, j]] = 1

    return train, test


class LSTM_Model():
    def __init__(self, input_shape, lr, attn_fusion=True, unimodal=False):
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
        self.attn_fusion = attn_fusion
        self.unimodal = unimodal
        self.dropout_keep_rate = tf.placeholder(tf.float32, name="dropout_keep_rate")
        # Build the model
        self._build_model_op()
        # self._build_loss()
        self._initialize_optimizer()

    def BiLSTM(self, inputs, output_size, name, dropout_keep_rate):
        with tf.variable_scope('rnn_' + name, reuse=tf.AUTO_REUSE):
            fw_cell = tf.contrib.rnn.GRUCell(output_size / 2, name='gru', reuse=tf.AUTO_REUSE)
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_rate)

            bw_cell = tf.contrib.rnn.GRUCell(output_size / 2, name='gru', reuse=tf.AUTO_REUSE)
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_rate)

            output_fw, _ = tf.nn.dynamic_rnn(fw_cell, inputs, sequence_length=self.seq_len, dtype=tf.float32)
            output_bw, _ = tf.nn.dynamic_rnn(bw_cell, inputs, sequence_length=self.seq_len, dtype=tf.float32)

            output = tf.concat([output_fw, output_bw], axis=-1)
            return output

    def self_attention(self, inputs_a, inputs_v, inputs_t, name, return_alphas=False):
        """
        inputs = (B, 3, T, dim)
        :param inputs:
        :param name:
        :param return_alphas:
        :return:
        """
        inputs_a = tf.expand_dims(inputs_a, axis=1)
        inputs_v = tf.expand_dims(inputs_v, axis=1)
        inputs_t = tf.expand_dims(inputs_t, axis=1)
        inputs = tf.concat([inputs_a, inputs_v, inputs_t], axis=1)
        t = inputs.get_shape()[2].value
        shared = True
        hidden_size = inputs.shape[-1].value  # D value - hidden size of the RNN layer
        if shared:
            scope_name = 'self_attn'
        else:
            scope_name = 'self_attn' + name
        print(scope_name)
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
                    # x_proj = inputs
                # print('x_proj', x_proj.get_shape())
                u_w = tf.Variable(tf.random_normal([hidden_size, 1], stddev=0.01, seed=1227))
                x = tf.tensordot(x_proj, u_w, axes=1)
                # print('x', x.get_shape())
                alphas = tf.nn.softmax(x, axis=-1)
                # print('alphas', alphas.get_shape())
                # output = tf.matmul(alphas, inputs)  # (B, T, 1) * (B, T, 1) => (B, T, D)
                output = tf.matmul(tf.transpose(t_x, [0, 2, 1]), alphas)
                output = tf.squeeze(output, -1)
                outputs.append(output)
                print('output', output.get_shape())

            final_output = tf.stack(outputs, axis=1)
            print('final_output', final_output.get_shape())
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

        lstm_output = self.BiLSTM(input, 300, 'lstm', self.dropout_keep_rate)
        inter = Dropout(self.dropout_keep_rate)(lstm_output)
        self.inter1 = Dense(100)(inter)
        inter = Dropout(self.dropout_keep_rate)(self.inter1)
        self.output = Dense(2)(inter)
        print('self.output', self.output.get_shape())
        self.preds = tf.nn.softmax(self.output)

        # To calculate the number correct, we want to count padded steps as incorrect
        correct = tf.cast(
            tf.equal(tf.argmax(self.preds, -1, output_type=tf.int32), tf.argmax(self.y, -1, output_type=tf.int32)),
            tf.int32) * tf.cast(self.mask, tf.int32)

        # To calculate accuracy we want to divide by the number of non-padded time-steps,
        # rather than taking the mean
        self.accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.reduce_sum(tf.cast(self.seq_len, tf.float32))
        print(self.output.shape)
        print(self.y.shape)
        y = tf.argmax(self.y, -1)
        print(y.shape)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=y)
        loss = loss * self.mask

        self.loss = tf.reduce_sum(loss) / tf.reduce_sum(self.mask)

    def _initialize_optimizer(self):
        self.global_step = tf.get_variable(shape=[], initializer=tf.constant_initializer(0), dtype=tf.int32,
                                           name='global_step')
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999)
        self.train_op = self._optimizer.minimize(self.loss, global_step=self.global_step)


def batch_iter(data, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


def multimodal(unimodal_activations):
    attn_fusion = True
    print("starting multimodal")
    # Fusion (appending) of features

    text_train = unimodal_activations['text_train']
    audio_train = unimodal_activations['audio_train']
    video_train = unimodal_activations['video_train']

    text_test = unimodal_activations['text_test']
    audio_test = unimodal_activations['audio_test']
    video_test = unimodal_activations['video_test']

    train_mask = unimodal_activations['train_mask']
    test_mask = unimodal_activations['test_mask']

    print('train_mask', train_mask.shape)

    train_label = unimodal_activations['train_label']
    print('train_label', train_label.shape)
    test_label = unimodal_activations['test_label']
    print('test_label', test_label.shape)

    # print(train_mask_bool)
    seqlen_train = np.sum(train_mask, axis=-1)
    print('seqlen_train', seqlen_train.shape)
    seqlen_test = np.sum(test_mask, axis=-1)
    print('seqlen_test', seqlen_test.shape)

    allow_soft_placement = True
    log_device_placement = False

    # Multimodal model
    session_conf = tf.ConfigProto(
        # device_count={'GPU': gpu_count},
        allow_soft_placement=allow_soft_placement,
        log_device_placement=log_device_placement,
        gpu_options=tf.GPUOptions(allow_growth=True))
    gpu_device = 0
    best_acc = 0
    best_epoch = 0
    with tf.device('/device:GPU:%d' % gpu_device):
        print('Using GPU - ', '/device:GPU:%d' % gpu_device)
        with tf.Graph().as_default():
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                model = LSTM_Model(text_train.shape[1:], 0.001, attn_fusion=attn_fusion, unimodal=False)
                sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

                test_feed_dict = {
                    model.t_input: text_test,
                    model.a_input: audio_test,
                    model.v_input: video_test,
                    model.y: test_label,
                    model.seq_len: seqlen_test,
                    model.mask: test_mask,
                    model.dropout_keep_rate: 1.0
                }
                # print('\n\nDataset: %s' % (data))
                print("\nEvaluation before training:")
                # Evaluation after epoch
                step, loss, accuracy = sess.run(
                    [model.global_step, model.loss, model.accuracy],
                    test_feed_dict)
                print("EVAL: epoch {}: step {}, loss {:g}, acc {:g}".format(0, step, loss, accuracy))

                epochs = 100
                batch_size = 10
                for epoch in range(epochs):
                    epoch += 1

                    batches = batch_iter(list(
                        zip(text_train, audio_train, video_train, train_mask, seqlen_train, train_label)),
                        batch_size)

                    # Training loop. For each batch...
                    print('\nTraining epoch {}'.format(epoch))
                    l = []
                    a = []
                    for i, batch in tqdm(enumerate(batches)):
                        b_text_train, b_audio_train, b_video_train, b_train_mask, b_seqlen_train, b_train_label = zip(
                            *batch)
                        # print('batch_hist_v', len(batch_utt_v))
                        feed_dict = {
                            model.t_input: b_text_train,
                            model.a_input: b_audio_train,
                            model.v_input: b_video_train,
                            model.y: b_train_label,
                            model.seq_len: b_seqlen_train,
                            model.mask: b_train_mask,
                            model.dropout_keep_rate: 0.6
                        }

                        _, step, loss, accuracy = sess.run(
                            [model.train_op, model.global_step, model.loss, model.accuracy],
                            feed_dict)
                        l.append(loss)
                        a.append(accuracy)

                    print("\t \tEpoch {}:, loss {:g}, acc/mae {:g}".format(epoch, np.average(l), np.average(a)))
                    # Evaluation after epoch
                    step, loss, accuracy = sess.run(
                        [model.global_step, model.loss, model.accuracy],
                        test_feed_dict)
                    print("EVAL: After epoch {}: step {}, loss {:g}, acc {:g}".format(epoch, step, loss, accuracy))

                    if accuracy > best_acc:
                        best_epoch = epoch
                        best_acc = accuracy

                print("\n\nBest epoch: {}\nBest test accuracy: {}".format(best_epoch, best_acc))


def unimodal(mode):
    print(('starting unimodal ', mode))

    with open('./input/' + mode + '.pickle', 'rb') as handle:
        u = pickle._Unpickler(handle)
        u.encoding = 'latin1'
        (train_data, train_label, test_data, test_label, maxlen, train_length, test_length) = u.load()

    # with open('./input/' + mode + '.pickle', 'rb') as handle:
    #     (train_data, train_label, test_data, test_label, maxlen, train_length, test_length) = pickle.load(handle)

    train_label = train_label.astype('int')
    test_label = test_label.astype('int')

    train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
    for i in range(len(train_length)):
        train_mask[i, :train_length[i]] = 1.0

    test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
    for i in range(len(test_length)):
        test_mask[i, :test_length[i]] = 1.0

    train_label, test_label = createOneHot(train_label, test_label)

    attn_fusion = False

    print('train_mask', train_mask.shape)

    # print(train_mask_bool)
    seqlen_train = np.sum(train_mask, axis=-1)
    print('seqlen_train', seqlen_train.shape)
    seqlen_test = np.sum(test_mask, axis=-1)
    print('seqlen_test', seqlen_test.shape)

    allow_soft_placement = True
    log_device_placement = False

    # Multimodal model
    session_conf = tf.ConfigProto(
        # device_count={'GPU': gpu_count},
        allow_soft_placement=allow_soft_placement,
        log_device_placement=log_device_placement,
        gpu_options=tf.GPUOptions(allow_growth=True))
    gpu_device = 0
    best_acc = 0
    best_epoch = 0
    is_unimodal = True
    with tf.device('/device:GPU:%d' % gpu_device):
        print('Using GPU - ', '/device:GPU:%d' % gpu_device)
        with tf.Graph().as_default():
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                model = LSTM_Model(train_data.shape[1:], 0.001, attn_fusion=attn_fusion, unimodal=is_unimodal)
                sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

                test_feed_dict = {
                    model.input: test_data,
                    model.y: test_label,
                    model.seq_len: seqlen_test,
                    model.mask: test_mask,
                    model.dropout_keep_rate: 1.0
                }
                train_feed_dict = {
                    model.input: train_data,
                    model.y: train_label,
                    model.seq_len: seqlen_train,
                    model.mask: train_mask,
                    model.dropout_keep_rate: 1.0
                }
                # print('\n\nDataset: %s' % (data))
                print("\nEvaluation before training:")
                # Evaluation after epoch
                step, loss, accuracy = sess.run(
                    [model.global_step, model.loss, model.accuracy],
                    test_feed_dict)
                print("EVAL: epoch {}: step {}, loss {:g}, acc {:g}".format(0, step, loss, accuracy))

                epochs = 100
                batch_size = 10
                for epoch in range(epochs):
                    epoch += 1

                    batches = batch_iter(list(
                        zip(train_data, train_mask, seqlen_train, train_label)),
                        batch_size)

                    # Training loop. For each batch...
                    print('\nTraining epoch {}'.format(epoch))
                    l = []
                    a = []
                    for i, batch in tqdm(enumerate(batches)):
                        b_train_data, b_train_mask, b_seqlen_train, b_train_label = zip(
                            *batch)
                        # print('batch_hist_v', len(batch_utt_v))
                        feed_dict = {
                            model.input: b_train_data,
                            model.y: b_train_label,
                            model.seq_len: b_seqlen_train,
                            model.mask: b_train_mask,
                            model.dropout_keep_rate: 0.6
                        }

                        _, step, loss, accuracy = sess.run(
                            [model.train_op, model.global_step, model.loss, model.accuracy],
                            feed_dict)
                        l.append(loss)
                        a.append(accuracy)

                    print("\t \tEpoch {}:, loss {:g}, acc/mae {:g}".format(epoch, np.average(l), np.average(a)))
                    # Evaluation after epoch
                    step, loss, accuracy, test_activations = sess.run(
                        [model.global_step, model.loss, model.accuracy, model.inter1],
                        test_feed_dict)
                    print("EVAL: After epoch {}: step {}, loss {:g}, acc {:g}".format(epoch, step, loss, accuracy))

                    if accuracy > best_acc:
                        best_epoch = epoch
                        best_acc = accuracy

                        step, loss, accuracy, train_activations = sess.run(
                            [model.global_step, model.loss, model.accuracy, model.inter1],
                            train_feed_dict)
                        unimodal_activations[mode + '_train'] = train_activations
                        unimodal_activations[mode + '_test'] = test_activations

                        unimodal_activations['train_mask'] = train_mask
                        unimodal_activations['test_mask'] = test_mask
                        unimodal_activations['train_label'] = train_label
                        unimodal_activations['test_label'] = test_label

                print("\n\nBest epoch: {}\nBest test accuracy: {}".format(best_epoch, best_acc))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--unimodal", type=str2bool, nargs='?',
                        const=True, default=False)
    args, _ = parser.parse_known_args(argv)

    if args.unimodal:

        print("Training unimodals first")

        modality = ['text', 'audio', 'video']
        for mode in modality:
            unimodal(mode)

        print("Saving unimodal activations")
        with open('unimodal.pickle', 'wb') as handle:
            pickle.dump(unimodal_activations, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('unimodal.pickle', 'rb') as handle:
    #     unimodal_activations = pickle.load(handle)

    with open('unimodal_old.pickle', 'rb') as handle:
        u = pickle._Unpickler(handle)
        u.encoding = 'latin1'
        unimodal_activations = u.load()

    multimodal(unimodal_activations)
