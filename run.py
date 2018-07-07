import argparse
import pickle
import sys

import numpy as np

seed = 1234

np.random.seed(seed)
import tensorflow as tf
from tqdm import tqdm

from model import LSTM_Model

tf.set_random_seed(seed)

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


def multimodal(unimodal_activations, attn_fusion=True):
    if attn_fusion:
        print('With attention fusion')
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
            tf.set_random_seed(seed)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                model = LSTM_Model(text_train.shape[1:], 0.001, attn_fusion=attn_fusion, unimodal=False, seed=seed)
                sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

                test_feed_dict = {
                    model.t_input: text_test,
                    model.a_input: audio_test,
                    model.v_input: video_test,
                    model.y: test_label,
                    model.seq_len: seqlen_test,
                    model.mask: test_mask,
                    model.lstm_dropout: 0.0,
                    model.dropout: 0.0
                }

                # print('\n\nDataset: %s' % (data))
                print("\nEvaluation before training:")
                # Evaluation after epoch
                step, loss, accuracy = sess.run(
                    [model.global_step, model.loss, model.accuracy],
                    test_feed_dict)
                print("EVAL: epoch {}: step {}, loss {:g}, acc {:g}".format(0, step, loss, accuracy))

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
                            model.lstm_dropout: 0.6,
                            model.dropout: 0.9
                        }

                        _, step, loss, accuracy = sess.run(
                            [model.train_op, model.global_step, model.loss, model.accuracy],
                            feed_dict)
                        l.append(loss)
                        a.append(accuracy)

                    print("\t \tEpoch {}:, loss {:g}, accuracy {:g}".format(epoch, np.average(l), np.average(a)))
                    # Evaluation after epoch
                    step, loss, accuracy = sess.run(
                        [model.global_step, model.loss, model.accuracy],
                        test_feed_dict)
                    print("EVAL: After epoch {}: step {}, loss {:g}, acc {:g}".format(epoch, step, loss/test_label.shape[0], accuracy))

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
                model = LSTM_Model(train_data.shape[1:], 0.001, attn_fusion=attn_fusion, unimodal=is_unimodal,
                                   seed=seed)
                sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

                test_feed_dict = {
                    model.input: test_data,
                    model.y: test_label,
                    model.seq_len: seqlen_test,
                    model.mask: test_mask,
                    model.lstm_dropout: 0.0,
                    model.dropout: 0.0
                }
                train_feed_dict = {
                    model.input: train_data,
                    model.y: train_label,
                    model.seq_len: seqlen_train,
                    model.mask: train_mask,
                    model.lstm_dropout: 0.0,
                    model.dropout: 0.0
                }
                # print('\n\nDataset: %s' % (data))
                print("\nEvaluation before training:")
                # Evaluation after epoch
                step, loss, accuracy = sess.run(
                    [model.global_step, model.loss, model.accuracy],
                    test_feed_dict)
                print("EVAL: epoch {}: step {}, loss {:g}, acc {:g}".format(0, step, loss, accuracy))

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
                            model.lstm_dropout: 0.6,
                            model.dropout: 0.9,
                        }

                        _, step, loss, accuracy = sess.run(
                            [model.train_op, model.global_step, model.loss, model.accuracy],
                            feed_dict)
                        l.append(loss)
                        a.append(accuracy)

                    print("\t \tEpoch {}:, loss {:g}, accuracy {:g}".format(epoch, np.average(l), np.average(a)))
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
    parser.add_argument("--unimodal", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--fusion", type=str2bool, nargs='?', const=True, default=False)
    args, _ = parser.parse_known_args(argv)
    batch_size = 10
    epochs = 60

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

    with open('unimodal.pickle', 'rb') as handle:
        u = pickle._Unpickler(handle)
        u.encoding = 'latin1'
        unimodal_activations = u.load()

    multimodal(unimodal_activations, args.fusion)
