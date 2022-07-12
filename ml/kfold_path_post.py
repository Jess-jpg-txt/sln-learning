import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Reshape, LSTM, InputLayer
import tensorflow_probability as tfp
import pandas as pd
from sklearn import metrics
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys

# load data
data_total = pd.read_csv('../data/w_removal_ml', sep=" ", header=None)
data_total.columns = ["n1", "n2", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "l"]

feats = 'path_post (f5,f6,f7)'

features = np.vstack([data_total["f5"], data_total["f6"], data_total["f7"]])
features = np.transpose(features)
labels = np.array(data_total["l"])

# reshape data
features = features.reshape(features.shape[0], features.shape[1], 1, 1)
labels_oh = keras.utils.to_categorical(labels, num_classes=2)

num_feat = features.shape[1]

# models
def cnn_model(num_feat):
    model = Sequential()
    model.add(Conv2D(64, (3, 1), activation='relu', use_bias=True, input_shape=(num_feat, 1, 1)))
    # model.add(Conv2D(32, (2, 1), activation='relu', use_bias=True))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', use_bias=True))
    model.add(Dense(2, activation='softmax', kernel_initializer='normal'))
    return model

def crnn_model(num_feat):
    model = Sequential()
    model.add(Conv2D(64, (3, 1), activation='relu', use_bias=True, input_shape=(num_feat, 1, 1)))
    # model.add(Conv2D(32, (2, 1), activation='relu', use_bias=True))
    model.add(Reshape((2, 32)))
    model.add(LSTM(32))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', use_bias=True))
    model.add(Dense(2, activation='softmax', kernel_initializer='normal'))
    return model

def rnn_model(num_feat):
    model = Sequential()
    model.add(LSTM(64, input_shape=(num_feat, 1)))
    model.add(Dense(128, activation='relu', kernel_initializer='normal', use_bias=True))
    model.add(Dense(2, activation='softmax', kernel_initializer='normal'))
    return model

def fcnn_model(num_feat):
    model = Sequential()
    model.add(Dense(128, activation='relu', kernel_initializer='normal', use_bias=True, input_dim=num_feat))
    model.add(Dense(128, activation='relu', kernel_initializer='normal', use_bias=True))
    model.add(Dense(2, activation='softmax', kernel_initializer='normal'))
    return model

def bnet_model(num_feat):
    encoded_size = 10
    n_classes = 2
    model = Sequential([InputLayer(input_shape=(num_feat,), name="input"),
                    Dense(tfp.layers.MultivariateNormalTriL.params_size(encoded_size), name = "shaping"),
                    tfp.layers.MultivariateNormalTriL(encoded_size, name="latent"),
                    Dense(n_classes, activation="softmax", name="output")])
    return model

# metrics to store acc and auc
acc_cnn = []
auc_cnn = []
acc_crnn = []
auc_crnn = []
acc_rnn = []
auc_rnn = []
acc_fcnn = []
auc_fcnn = []
acc_bnet = []
auc_bnet = []
acc_svm = []
auc_svm = []
acc_lda = []
auc_lda = []
acc_baseline = []
auc_baseline = []

# define the K-fold cross validator
kfold = KFold(n_splits=10, shuffle=True)

# train the model
fold_no = 1
epochs = 300
batch_size = 64
for train, test in kfold.split(features, labels_oh):

    print('Training for fold %d' % fold_no)

    training_samples = features[train]
    testing_samples = features[test]

    # cnn cv
    cnn = cnn_model(num_feat=num_feat)
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn.fit(training_samples, labels_oh[train], epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    # calculate and collect model accuracy
    scores = cnn.evaluate(testing_samples, labels_oh[test], verbose=1)
    acc_cnn.append(scores[1])
    # calculate and collect model AUC
    predicted_classes = np.argmax(cnn.predict(testing_samples), axis=-1)
    fpr, tpr, _ = metrics.roc_curve(labels[test], predicted_classes, pos_label=1)
    auc_cnn.append(metrics.auc(fpr, tpr))

    # crnn cv
    crnn = crnn_model(num_feat=num_feat)
    crnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    crnn.fit(training_samples, labels_oh[train], epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    # calculate and collect model accuracy
    scores = crnn.evaluate(testing_samples, labels_oh[test], verbose=1)
    acc_crnn.append(scores[1])
    # calculate and collect model AUC
    predicted_classes = np.argmax(crnn.predict(testing_samples), axis=-1)
    fpr, tpr, _ = metrics.roc_curve(labels[test], predicted_classes, pos_label=1)
    auc_crnn.append(metrics.auc(fpr, tpr))

    # rnn cv
    training_samples = training_samples.reshape(training_samples.shape[0], training_samples.shape[1], training_samples.shape[2])
    testing_samples = testing_samples.reshape(testing_samples.shape[0], testing_samples.shape[1], testing_samples.shape[2])
    rnn = rnn_model(num_feat=num_feat)
    rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    rnn.fit(training_samples, labels_oh[train], epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    # calculate and collect model accuracy
    scores = rnn.evaluate(testing_samples, labels_oh[test], verbose=1)
    acc_rnn.append(scores[1])
    # calculate and collect model AUC
    predicted_classes = np.argmax(rnn.predict(testing_samples), axis=-1)
    fpr, tpr, _ = metrics.roc_curve(labels[test], predicted_classes, pos_label=1)
    auc_rnn.append(metrics.auc(fpr, tpr))

    # fcnn cv
    training_samples = training_samples.reshape(training_samples.shape[0], training_samples.shape[1]*training_samples.shape[2])
    testing_samples = testing_samples.reshape(testing_samples.shape[0], testing_samples.shape[1]*testing_samples.shape[2])
    fcnn = fcnn_model(num_feat=num_feat)
    fcnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    fcnn.fit(training_samples, labels_oh[train], epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    # calculate and collect model accuracy
    scores = fcnn.evaluate(testing_samples, labels_oh[test], verbose=1)
    acc_fcnn.append(scores[1])
    # calculate and collect model AUC
    predicted_classes = np.argmax(fcnn.predict(testing_samples), axis=-1)
    fpr, tpr, _ = metrics.roc_curve(labels[test], predicted_classes, pos_label=1)
    auc_fcnn.append(metrics.auc(fpr, tpr))

    # bnet cv
    bnet = bnet_model(num_feat=num_feat)
    bnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    bnet.fit(training_samples, labels_oh[train], epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    # calculate and collect model accuracy
    scores = bnet.evaluate(testing_samples, labels_oh[test], verbose=1)
    acc_bnet.append(scores[1])
    # calculate and collect model AUC
    predicted_classes = np.argmax(bnet.predict(testing_samples), axis=-1)
    fpr, tpr, _ = metrics.roc_curve(labels[test], predicted_classes, pos_label=1)
    auc_bnet.append(metrics.auc(fpr, tpr))

    # svm
    clf_svm = svm.SVC(kernel='linear')
    clf_svm.fit(training_samples, labels[train])
    y_pred_test_svm = clf_svm.predict(testing_samples)
    # accuracy
    acc_svm.append(np.sum(y_pred_test_svm == labels[test]) / labels[test].shape[0])
    # auc
    fpr_svm, tpr_svm, _ = metrics.roc_curve(labels[test], y_pred_test_svm, pos_label=1)
    auc_svm.append(metrics.auc(fpr_svm, tpr_svm))

    # lda
    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(training_samples, labels[train])
    y_pred_test_lda = clf_lda.predict(testing_samples)
    #accuracy
    acc_lda.append(np.sum(y_pred_test_lda == labels[test]) / labels[test].shape[0])
    #auc
    fpr_lda, tpr_lda, _ = metrics.roc_curve(labels[test], y_pred_test_lda, pos_label=1)
    auc_lda.append(metrics.auc(fpr_lda, tpr_lda))

    # baseline
    re = testing_samples[:, 2]
    norm_factor = np.max(re)
    re = re / norm_factor
    bl_predictions = np.zeros((re.shape[0]))
    ones_idx = np.where(re >= 0.5)
    bl_predictions[ones_idx] = 1
    # accuracy
    acc_baseline.append(np.sum(bl_predictions == labels[test]) / labels[test].shape[0])
    # auc
    fpr_bl, tpr_bl, _ = metrics.roc_curve(labels[test], bl_predictions, pos_label=1)
    auc_baseline.append(metrics.auc(fpr_bl, tpr_bl))

    fold_no = fold_no + 1

acc_cnn = np.array(acc_cnn)
auc_cnn = np.array(auc_cnn)
acc_crnn = np.array(acc_crnn)
auc_crnn = np.array(auc_crnn)
acc_rnn = np.array(acc_rnn)
auc_rnn = np.array(auc_rnn)
acc_fcnn = np.array(acc_fcnn)
auc_fcnn = np.array(auc_fcnn)
acc_bnet = np.array(acc_bnet)
auc_bnet = np.array(auc_bnet)
acc_svm = np.array(acc_svm)
auc_svm = np.array(auc_svm)
acc_lda = np.array(acc_lda)
auc_lda = np.array(auc_lda)
acc_baseline = np.array(acc_baseline)
auc_baseline = np.array(auc_baseline)

sys.stdout = open("kfold_results.txt", "a")
print("************************" + feats + "***********************")

print('CNN Accuracy: {} +/- {}'.format(np.mean(acc_cnn), np.std(acc_cnn)))
print('CNN AUC: {} +/- {}'.format(np.mean(auc_cnn), np.std(auc_cnn)))
print('CRNN Accuracy: {} +/- {}'.format(np.mean(acc_crnn), np.std(acc_crnn)))
print('CRNN AUC: {} +/- {}'.format(np.mean(auc_crnn), np.std(auc_crnn)))
print('RNN Accuracy: {} +/- {}'.format(np.mean(acc_rnn), np.std(acc_rnn)))
print('RNN AUC: {} +/- {}'.format(np.mean(auc_rnn), np.std(auc_rnn)))
print('FCNN Accuracy: {} +/- {}'.format(np.mean(acc_fcnn), np.std(acc_fcnn)))
print('FCNN AUC: {} +/- {}'.format(np.mean(auc_fcnn), np.std(auc_fcnn)))
print('BNet Accuracy: {} +/- {}'.format(np.mean(acc_bnet), np.std(acc_bnet)))
print('BNet AUC: {} +/- {}'.format(np.mean(auc_bnet), np.std(auc_bnet)))
print('SVM Accuracy: {} +/- {}'.format(np.mean(acc_svm), np.std(acc_svm)))
print('SVM AUC: {} +/- {}'.format(np.mean(auc_svm), np.std(auc_svm)))
print('LDA Accuracy: {} +/- {}'.format(np.mean(acc_lda), np.std(acc_lda)))
print('LDA AUC: {} +/- {}'.format(np.mean(auc_lda), np.std(auc_lda)))
print('Baseline Accuracy: {} +/- {}'.format(np.mean(acc_baseline), np.std(acc_baseline)))
print('Baseline AUC: {} +/- {}'.format(np.mean(auc_baseline), np.std(auc_baseline)))

sys.stdout.close()

