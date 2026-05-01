# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 07:03:05 2022

@author: gastong@fing.edu.uy
"""


from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv1D, BatchNormalization, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.keras.regularizers import l2
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score
import pickle
from tqdm import tqdm

import numpy as np
from itertools import groupby
from operator import itemgetter

from spaceai.models.legacy.anomaly_classifier import AnomalyClassifier


def interval_intersection(I=(1, 3), J=(2, 4)):
    """
    Intersection between two intervals I and J
    I and J should be either empty or represent a positive interval (no point)

    :param I: an interval represented by start and stop
    :param J: a second interval of the same form
    :return: an interval representing the start and stop of the intersection (or None if empty)
    """
    if I is None:
        return (None)
    if J is None:
        return (None)

    I_inter_J = (max(I[0], J[0]), min(I[1], J[1]))
    if I_inter_J[0] >= I_inter_J[1]:
        return (None)
    else:
        return (I_inter_J)


def convert_vector_to_events(vector=[0, 1, 1, 0, 0, 1, 0]):
    """
    Convert a binary vector (indicating 1 for the anomalous instances)
    to a list of events. The events are considered as durations,
    i.e. setting 1 at index i corresponds to an anomalous interval [i, i+1).

    :param vector: a list of elements belonging to {0, 1}
    :return: a list of couples, each couple representing the start and stop of
    each event
    """
    positive_indexes = [idx for idx, val in enumerate(vector) if val > 0]
    events = []
    for k, g in groupby(enumerate(positive_indexes), lambda ix: ix[0] - ix[1]):
        cur_cut = list(map(itemgetter(1), g))
        events.append((cur_cut[0], cur_cut[-1]))

    # Consistent conversion in case of range anomalies (for indexes):
    # A positive index i is considered as the interval [i, i+1),
    # so the last index should be moved by 1
    events = [(x, y + 1) for (x, y) in events]

    return events


def per_event_f_score(y_true: np.ndarray, y_score: np.ndarray, beta=1.0):
    y_pred = y_score

    events_pred = convert_vector_to_events(y_pred)  # [(4, 5), (8, 9)]
    events_gt = convert_vector_to_events(y_true)  # [(3, 4), (7, 10)]

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_events_pred = [False for _ in events_pred]
    for gt in events_gt:
        for p, pred in enumerate(events_pred):
            if pred[0] > gt[1]:
                break
            if interval_intersection(pred, gt) is None:
                continue
            matched_events_pred[p] = True
            true_positives += 1
            break
        else:
            false_negatives += 1

    for p, pred in enumerate(events_pred):
        if matched_events_pred[p]:
            continue
        for gt in events_gt:
            if gt[0] > pred[1]:
                false_positives += 1
                break
            if interval_intersection(pred, gt) is not None:
                break
        else:
            false_positives += 1

    if true_positives + false_positives == 0:
        precision = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)
        # Correction from (Sehili et al., 2023) http://arxiv.org/abs/2308.13068
        nominal_samples = (y_true == 0)
        detected_samples = (y_pred == 1)
        false_positive_samples = (detected_samples & nominal_samples)
        precision *= (1 - sum(false_positive_samples) / sum(nominal_samples))

    if true_positives + false_negatives == 0:
        recall = 0.0
    else:
        recall = true_positives / (true_positives + false_negatives)

    divider = beta ** 2 * precision + recall
    if divider == 0.0:
        result = 0.0
    else:
        result = ((1 + beta ** 2) * precision * recall) / divider

    return result


@keras.utils.register_keras_serializable()
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, name=None, k=1, **kwargs):
        super(Sampling, self).__init__(name=name)
        self.k = k
        super(Sampling, self).__init__(**kwargs)

    def get_config(self):
        config = super(Sampling, self).get_config()
        config['k'] = self.k
        return config  # dict(list(config.items()))

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        seq = K.shape(z_mean)[1]
        dim = K.shape(z_mean)[2]
        epsilon = K.random_normal(shape=(batch, seq, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


@keras.utils.register_keras_serializable()
class DCVAELoss(Layer):
    """Adds DC-VAE objective terms to the model via Layer.add_loss."""

    def __init__(self, variance_clip=10.0, **kwargs):
        super().__init__(**kwargs)
        self.variance_clip = variance_clip

    def get_config(self):
        config = super().get_config()
        config["variance_clip"] = self.variance_clip
        return config

    def call(self, inputs):
        outputs, x__mean, x_log_var, z_mean, z_log_var = inputs

        # Keep the variance terms in a numerically stable range.
        x_log_var = keras.ops.clip(
            x_log_var, -self.variance_clip, self.variance_clip)
        z_log_var = keras.ops.clip(
            z_log_var, -self.variance_clip, self.variance_clip)

        # Reconstruction term
        mse = -0.5 * keras.ops.mean(
            keras.ops.square((outputs - x__mean) / keras.ops.exp(x_log_var)),
            axis=-1,
        )
        sigma_trace = -keras.ops.mean(x_log_var, axis=-1)
        log_likelihood = mse + sigma_trace
        reconstruction_loss = keras.ops.mean(-log_likelihood)

        # Prior term
        kl_loss = 1 + z_log_var - \
            keras.ops.square(z_mean) - keras.ops.exp(z_log_var)
        kl_loss = keras.ops.mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        kl_loss = keras.ops.mean(kl_loss)

        # Total objective registered on the layer/model
        vae_loss = reconstruction_loss + kl_loss
        self.add_loss(vae_loss)

        return x__mean, x_log_var


class DCVAE:

    def __init__(self,
                 M_output=1,
                 T=50,
                 M=1,
                 cnn_units=[32, 16, 1],
                 dil_rate=[1, 8, 16],
                 kernel=2,
                 strs=1,
                 batch_size=32,
                 J=1,
                 epochs=100,
                 learning_rate=1e-3,
                 lr_decay=True,
                 decay_rate=0.96,
                 decay_step=1000,
                 name='',
                 epsilon=1e-12,
                 summary=True,
                 ):

        # network parameters
        input_shape = (T, M)
        self.M = M
        self.M_output = M_output
        self.T = T
        self.J = J
        self.batch_size = batch_size
        self.epochs = epochs
        self.name = name

        # model = encoder + decoder

        # Build encoder model
        # =============================================================================
        # Input
        inputs = Input(shape=input_shape, name='input')
        outputs = Input(shape=(T, M_output), name="output")

        # Hidden layers (1D Dilated Convolution)
        # First
        h_enc_cnn = Conv1D(cnn_units[0], kernel, activation='tanh', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001),
                           strides=strs, padding="causal",
                           dilation_rate=dil_rate[0], name='cnn_%d' % 0)(inputs)
        h_enc_cnn = BatchNormalization()(h_enc_cnn)

        # Middle
        for i in range(len(cnn_units)-2):
            h_enc_cnn = Conv1D(cnn_units[i+1], kernel, activation='tanh', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001),
                               strides=strs, padding="causal",
                               dilation_rate=dil_rate[i+1], name='cnn_%d' % (i+1))(h_enc_cnn)
            h_enc_cnn = BatchNormalization()(h_enc_cnn)

        # Lastest
        z_mean = Conv1D(J, kernel, activation=None,
                        strides=strs, padding="causal",
                        dilation_rate=dil_rate[i+1], name='z_mean')(h_enc_cnn)
        z_log_var = Conv1D(J, kernel, activation=None,
                           strides=strs, padding="causal",
                           dilation_rate=dil_rate[i+1], name='z_log_var')(h_enc_cnn)

        # Reparameterization trick
        # Output
        z = Sampling(name='z')((z_mean, z_log_var))
        # Instantiate encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        if summary:
            self.encoder.summary()
        # =============================================================================

        # Build decoder model
        # =============================================================================
        # Input
        latent_inputs = Input(shape=(T, J), name='z_sampling')

        # Hidden layers (1D Dilated Convolution)
        # First
        h_dec_cnn = Conv1D(cnn_units[-1], kernel, activation='elu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001),
                           strides=strs, padding="causal",
                           dilation_rate=dil_rate[-1], name='cnn_-1')(latent_inputs)
        h_dec_cnn = BatchNormalization()(h_dec_cnn)

        # Middle
        for i in range(-2, -len(cnn_units), -1):
            h_dec_cnn = Conv1D(cnn_units[i], kernel, activation='elu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001),
                               strides=strs, padding="causal",
                               dilation_rate=dil_rate[i], name='cnn_%d' % i)(h_dec_cnn)
            h_dec_cnn = BatchNormalization()(h_dec_cnn)

        # Lastest/Output
        x__mean = Conv1D(M_output, kernel, activation=None,
                         padding="causal",
                         dilation_rate=dil_rate[0],
                         name='x__mean_output')(h_dec_cnn)
        x_log_var = Conv1D(M_output, kernel, activation=None,
                           padding="causal",
                           dilation_rate=dil_rate[0],
                           name='x_log_var_output')(h_dec_cnn)

        # Instantiate decoder model
        self.decoder = Model(
            latent_inputs, [x__mean, x_log_var], name='decoder')
        if summary:
            self.decoder.summary()
        # =============================================================================

        # Instantiate DC-VAE model
        # =============================================================================
        [x__mean, x_log_var] = self.decoder(self.encoder(inputs)[2])

        # Attach the objective before building the final model so Keras tracks it.
        x__mean, x_log_var = DCVAELoss(name='vae_loss')(
            [outputs, x__mean, x_log_var, z_mean, z_log_var]
        )

        self.vae = Model([inputs, outputs], [x__mean, x_log_var], name='vae')

        # Learning rate
        if lr_decay:
            lr = optimizers.schedules.ExponentialDecay(learning_rate,
                                                       decay_steps=decay_step,
                                                       decay_rate=decay_rate,
                                                       staircase=True,
                                                       )
        else:
            lr = learning_rate

        # Optimaizer
        opt = optimizers.Adam(learning_rate=lr)

        self.vae.compile(optimizer=opt)

    def _prepare_vae_inputs(self, X):
        """Build the 2-input structure expected by the VAE."""
        X = np.asarray(X)
        if X.ndim == 2:
            X = X[..., np.newaxis]
        X = X.astype(np.float32)
        train_inputs = (
            X,
            X[..., :self.M_output],
        )
        return train_inputs

    def fit(self, channel, model_path: Optional[str] = None):
        callbacks = []
        # Callbacks
        early_stopping_cb = keras.callbacks.EarlyStopping(min_delta=1e-3,
                                                          patience=20,
                                                          verbose=1,
                                                          mode='min')
        callbacks.append(early_stopping_cb)
        if model_path is not None:
            model_checkpoint_cb = keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                verbose=1,
                mode='min',
                save_best_only=True)
            callbacks.append(model_checkpoint_cb)

        # Model train
        self.max_steps_per_epoch = 1000
        train_inputs = self._prepare_vae_inputs(channel)
        steps_per_epoch = None
        if not isinstance(channel, np.ndarray):
            steps_per_epoch = min(self.max_steps_per_epoch, len(channel))

        self.history_ = self.vae.fit(
            train_inputs,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=True
        )

        if model_path is not None:
            pd.DataFrame.from_dict(self.history_.history).to_csv(
                model_path + '_history.csv', index=False)

            # Save models
            self.encoder.save(model_path+'_encoder.h5')
            self.decoder.save(model_path+'_decoder.h5')
            self.vae.save(model_path+'_complete.h5')

        return self

    def alpha_selection(self, X, y, model_path: Optional[str] = None, load_model=False, custom_metrics=False):

        # Model
        if model_path is not None and load_model:
            self.vae = keras.models.load_model(model_path,
                                               custom_objects={
                                                   'sampling': Sampling},
                                               compile=False)

        # Inference model. Auxiliary model so that in the inference
        # the prediction is only the last value of the sequence
        inp = Input(shape=(self.T, self.M))
        output = Input(shape=(self.T, self.M_output))
        x = self.vae([inp, output])  # apply trained model on the input
        out = Lambda(lambda y: [y[0][:, -1, :], y[1][:, -1, :]])(x)
        inference_model = Model([inp, output], out)

        # Predict
        batch = X[..., np.newaxis]
        prediction = inference_model([batch, batch[..., :self.M_output]])
        reconstruct = prediction[0].numpy()
        sig = prediction[1].numpy()
        sig = np.sqrt(np.exp(sig))

        # Data evaluate (The first T-1 data are discarded)
        X_evaluate = X[:, self.T - 1:]
        y_evaluate = y  # [self.T - 1:]

        # Threshold selection
        best_f1 = np.zeros(self.M_output)
        max_alpha = 7
        best_alpha_up = max_alpha * np.ones(self.M_output)
        best_alpha_down = max_alpha * np.ones(self.M_output)

        for alpha_up in np.arange(max_alpha, 1, -1):
            for alpha_down in np.arange(max_alpha, 1, -1):

                pre_predict = (X_evaluate < reconstruct - alpha_down * sig) | (
                    X_evaluate > reconstruct + alpha_up * sig)
                pre_predict = pre_predict.astype(int)

                for c in range(self.M_output):
                    if custom_metrics:
                        f1_value = per_event_f_score(
                            y_evaluate, pre_predict[:, c], beta=0.5)
                    else:
                        f1_value = fbeta_score(
                            y_evaluate, pre_predict[:, c], beta=1.0)

                    if f1_value >= best_f1[c]:
                        best_f1[c] = f1_value
                        best_alpha_up[c] = alpha_up
                        best_alpha_down[c] = alpha_down

        self.alpha_up = best_alpha_up
        self.alpha_down = best_alpha_down
        self.f1_val = best_f1

        if model_path is not None:
            with open(model_path + '_alpha_up.pkl', 'wb') as f:
                pickle.dump(best_alpha_up, f)
                f.close()
            with open(model_path + '_alpha_down.pkl', 'wb') as f:
                pickle.dump(best_alpha_down, f)
                f.close()

        return self

    def predict(self, channel,
                model_path: Optional[str] = None,
                load_model=False,
                load_alpha=True,
                alpha_set_up=[],
                alpha_set_down=[]):

        # Trained model
        if model_path is not None and load_model:
            self.vae = keras.models.load_model(model_path,
                                               custom_objects={
                                                   'sampling': Sampling},
                                               compile=False)

        # Inference model. Auxiliary model so that in the inference
        # the prediction is only the last value of the sequence
        inp = Input(shape=(self.T, self.M))
        output = Input(shape=(self.T, self.M_output))
        x = self.vae([inp, output])  # apply trained model on the input
        out = Lambda(lambda y: [y[0][:, -1, :], y[1][:, -1, :]])(x)
        inference_model = Model([inp, output], out)

        if isinstance(channel, np.ndarray):
            X = np.asarray(channel)
            if X.ndim == 2:
                X = X[..., np.newaxis]
            X = X.astype(np.float32)

            reconstruct, log_var = inference_model.predict(
                [X, X[..., :self.M_output]],
                verbose=0
            )
            sig = np.sqrt(np.exp(log_var))
            X_evaluate = X[:, -1, :self.M_output]
        else:
            reconstruct = []
            sig = []
            for i in tqdm(range(len(channel))):
                sample = channel[i]
                if isinstance(sample, (list, tuple)) and len(sample) == 2:
                    prediction = inference_model(sample)
                else:
                    prediction = inference_model(
                        [sample, sample[..., :self.M_output]]
                    )
                reconstruct.append(prediction[0].numpy())
                sig.append(np.sqrt(np.exp(prediction[1].numpy())))

            reconstruct = np.concatenate(reconstruct)
            sig = np.concatenate(sig)

            # Revert standardization only when channel metadata is available.
            if hasattr(channel, 'generator_test'):
                reconstruct = reconstruct * \
                    channel.generator_test.train_stds[channel] + \
                    channel.generator_test.train_means[channel]
                sig *= channel.generator_test.train_stds[channel]

            X_evaluate = channel[0, self.T - 1:]

        # Thresholds
        if len(alpha_set_up) == self.M_output:
            alpha_up = np.array(alpha_set_up)
        elif load_alpha and model_path is not None:
            with open(model_path + '_alpha_up.pkl', 'rb') as f:
                alpha_up = pickle.load(f)
        else:
            alpha_up = self.alpha_up

        if len(alpha_set_down) == self.M_output:
            alpha_down = np.array(alpha_set_down)
        elif load_alpha and model_path is not None:
            with open(model_path + '_alpha_down.pkl', 'rb') as f:
                alpha_down = pickle.load(f)
        else:
            alpha_down = self.alpha_down

        thdown = reconstruct - alpha_down*sig
        thup = reconstruct + alpha_up*sig

        # Evaluation
        pred = (X_evaluate < thdown) | (X_evaluate > thup)
        return pred.astype(int)


class DCVAEClassifier(AnomalyClassifier):
    def __init__(self, *args, alpha: Optional[int] = None, **kwargs):
        """
        Anomaly classifier based on DC-VAE. The model is trained on the input channel and then used to predict anomalies on the same channel.

        The parameter T is the length of the input sequences used for training and prediction.
        T is in kwargs and should be set according to the window length.

        :param args: Positional arguments (not used)
        :param alpha: Optional alpha value for thresholding. If not provided, it will be selected based on the training data.
        :param kwargs: Keyword arguments for DCVAE model initialization
        """
        super().__init__(*args, **kwargs)
        self.model = DCVAE(*args, **kwargs)
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        self.model.fit(X)
        if self.alpha is not None:
            self.model.alpha_up = np.array([self.alpha] * self.model.M_output)
            self.model.alpha_down = np.array(
                [self.alpha] * self.model.M_output)
        else:
            self.model.alpha_selection(X, y)

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        pred = self.model.predict(X)
        return pred

    def prepare_labels(self, channel_labels):
        return channel_labels
