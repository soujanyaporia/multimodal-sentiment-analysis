import pickle

import numpy as np


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


def createOneHotMosei3way(train_label, test_label):
    maxlen = 2
    # print(maxlen)

    train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            if train_label[i, j] > 0:
                train[i, j, 1] = 1
            else:
                if train_label[i, j] < 0:
                    train[i, j, 0] = 1
                else:
                    if train_label[i, j] == 0:
                        train[i, j, 2] = 1

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            if test_label[i, j] > 0:
                test[i, j, 1] = 1
            else:
                if test_label[i, j] < 0:
                    test[i, j, 0] = 1
                else:
                    if test_label[i, j] == 0:
                        test[i, j, 2] = 1
    return train, test


def createOneHotMosei2way(train_label, test_label):
    maxlen = 1
    # print(maxlen)

    train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            if train_label[i, j] > 0:
                train[i, j, 1] = 1
            else:
                if train_label[i, j] <= 0:
                    train[i, j, 0] = 1

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            if test_label[i, j] > 0:
                test[i, j, 1] = 1
            else:
                if test_label[i, j] <= 0:
                    test[i, j, 0] = 1

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


def prepare_iemocap(pyver):
    if pyver == 2:
        f = open("input/IEMOCAP_features/IEMOCAP_features.pkl", "rb")
        videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = pickle.load(
            f)

        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
    else:
        f = open("input/IEMOCAP_features/IEMOCAP_features.pkl", "rb")
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = u.load()
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

    print(len(trainVid))
    print(len(testVid))
    # exit(0)
    emotions = []

    train_audio = []
    train_text = []
    train_visual = []
    train_seq_len = []
    train_label = []
    train_mask = []

    test_audio = []
    test_text = []
    test_visual = []
    test_seq_len = []
    test_label = []
    train_mask = []
    max_len = 0
    for vid in trainVid:
        train_seq_len.append(len(videoIDs[vid]))
    for vid in testVid:
        test_seq_len.append(len(videoIDs[vid]))

    max_len = max(max(train_seq_len), max(test_seq_len))
    print('max_len', max_len)
    for vid in trainVid:
        train_label.append(videoLabels[vid])
        pad = [np.zeros(videoText[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        text = np.stack(videoText[vid] + pad, axis=0)
        train_text.append(text)

        pad = [np.zeros(videoAudio[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        audio = np.stack(videoAudio[vid] + pad, axis=0)
        train_audio.append(audio)

        pad = [np.zeros(videoVisual[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        video = np.stack(videoVisual[vid] + pad, axis=0)
        train_visual.append(video)

    for vid in testVid:
        test_label.append(videoLabels[vid])
        pad = [np.zeros(videoText[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        text = np.stack(videoText[vid] + pad, axis=0)
        test_text.append(text)

        pad = [np.zeros(videoAudio[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        audio = np.stack(videoAudio[vid] + pad, axis=0)
        test_audio.append(audio)

        pad = [np.zeros(videoVisual[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        video = np.stack(videoVisual[vid] + pad, axis=0)
        test_visual.append(video)

    train_text = np.stack(train_text, axis=0)
    train_audio = np.stack(train_audio, axis=0)
    train_visual = np.stack(train_visual, axis=0)
    print(train_text.shape)
    print(train_audio.shape)
    print(train_visual.shape)

    print()
    test_text = np.stack(test_text, axis=0)
    test_audio = np.stack(test_audio, axis=0)
    test_visual = np.stack(test_visual, axis=0)
    print(test_text.shape)
    print(test_audio.shape)
    print(test_visual.shape)

    print(train_label)
    print(train_seq_len)
    train_label = np.array(train_label)
    test_label = np.array(train_label)
    train_seq_len = np.array(train_seq_len)
    test_seq_len = np.array(test_seq_len)

    return train_audio, train_text, train_visual, test_audio, test_text, test_visual, train_label, test_label, train_seq_len, test_seq_len
