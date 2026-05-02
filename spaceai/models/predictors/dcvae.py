# -*- coding: utf-8 -*-
from typing import Optional, Any, List, Union, Tuple, Dict
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv1D, BatchNormalization, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow import keras
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from itertools import groupby
from operator import itemgetter
from sklearn.metrics import fbeta_score

def interval_intersection(I=(1, 3), J=(2, 4)):
    if I is None or J is None:
        return None
    I_inter_J = (max(I[0], J[0]), min(I[1], J[1]))
    if I_inter_J[0] >= I_inter_J[1]:
        return None
    else:
        return I_inter_J

def convert_vector_to_events(vector=[0, 1, 1, 0, 0, 1, 0]):
    positive_indexes = [idx for idx, val in enumerate(vector) if val > 0]
    events = []
    for k, g in groupby(enumerate(positive_indexes), lambda ix: ix[0] - ix[1]):
        cur_cut = list(map(itemgetter(1), g))
        events.append((cur_cut[0], cur_cut[-1]))
    events = [(x, y + 1) for (x, y) in events]
    return events

def per_event_f_score(y_true: np.ndarray, y_score: np.ndarray, beta=1.0):
    y_pred = y_score
    events_pred = convert_vector_to_events(y_pred)
    events_gt = convert_vector_to_events(y_true)

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
        nominal_samples = (y_true == 0)
        detected_samples = (y_pred == 1)
        false_positive_samples = (detected_samples & nominal_samples)
        if sum(nominal_samples) > 0:
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
    def __init__(self, name=None, k=1, **kwargs):
        super(Sampling, self).__init__(name=name)
        self.k = k
        super(Sampling, self).__init__(**kwargs)

    def get_config(self):
        config = super(Sampling, self).get_config()
        config['k'] = self.k
        return config

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        seq = K.shape(z_mean)[1]
        dim = K.shape(z_mean)[2]
        epsilon = K.random_normal(shape=(batch, seq, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

@keras.utils.register_keras_serializable()
class DCVAELoss(Layer):

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
        kl_loss = 1 + z_log_var - keras.ops.square(z_mean) - keras.ops.exp(z_log_var)
        kl_loss = keras.ops.mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        kl_loss = keras.ops.mean(kl_loss)
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
        input_shape = (T, M)
        self.M = M
        self.M_output = M_output
        self.T = T
        self.J = J
        self.batch_size = batch_size
        self.epochs = epochs
        self.name = name

        inputs = Input(shape=input_shape, name='input')
        outputs = Input(shape=(T, M_output), name="output")

        h_enc_cnn = Conv1D(cnn_units[0], kernel, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.001), 
                           bias_regularizer=keras.regularizers.l2(0.001), strides=strs, padding="causal",
                           dilation_rate=dil_rate[0], name='cnn_%d' % 0)(inputs)
        h_enc_cnn = BatchNormalization()(h_enc_cnn)

        for i in range(len(cnn_units)-2):
            h_enc_cnn = Conv1D(cnn_units[i+1], kernel, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.001),
                               bias_regularizer=keras.regularizers.l2(0.001), strides=strs, padding="causal",
                               dilation_rate=dil_rate[i+1], name='cnn_%d' % (i+1))(h_enc_cnn)
            h_enc_cnn = BatchNormalization()(h_enc_cnn)

        z_mean = Conv1D(J, kernel, activation=None, strides=strs, padding="causal",
                        dilation_rate=dil_rate[i+1], name='z_mean')(h_enc_cnn)
        z_log_var = Conv1D(J, kernel, activation=None, strides=strs, padding="causal",
                           dilation_rate=dil_rate[i+1], name='z_log_var')(h_enc_cnn)

        z = Sampling(name='z')((z_mean, z_log_var))
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        
        latent_inputs = Input(shape=(T, J), name='z_sampling')
        h_dec_cnn = Conv1D(cnn_units[-1], kernel, activation='elu', kernel_regularizer=keras.regularizers.l2(0.001),
                           bias_regularizer=keras.regularizers.l2(0.001), strides=strs, padding="causal",
                           dilation_rate=dil_rate[-1], name='cnn_-1')(latent_inputs)
        h_dec_cnn = BatchNormalization()(h_dec_cnn)

        for i in range(-2, -len(cnn_units), -1):
            h_dec_cnn = Conv1D(cnn_units[i], kernel, activation='elu', kernel_regularizer=keras.regularizers.l2(0.001),
                               bias_regularizer=keras.regularizers.l2(0.001), strides=strs, padding="causal",
                               dilation_rate=dil_rate[i], name='cnn_%d' % i)(h_dec_cnn)
            h_dec_cnn = BatchNormalization()(h_dec_cnn)

        x__mean = Conv1D(M_output, kernel, activation=None, padding="causal", dilation_rate=dil_rate[0], name='x__mean_output')(h_dec_cnn)
        x_log_var = Conv1D(M_output, kernel, activation=None, padding="causal", dilation_rate=dil_rate[0], name='x_log_var_output')(h_dec_cnn)

        self.decoder = Model(latent_inputs, [x__mean, x_log_var], name='decoder')
        [x__mean, x_log_var] = self.decoder(self.encoder(inputs)[2])
        x__mean, x_log_var = DCVAELoss(name='vae_loss')([outputs, x__mean, x_log_var, z_mean, z_log_var])
        self.vae = Model([inputs, outputs], [x__mean, x_log_var], name='vae')

        if lr_decay:
            lr = optimizers.schedules.ExponentialDecay(learning_rate, decay_steps=decay_step, decay_rate=decay_rate, staircase=True)
        else:
            lr = learning_rate
        self.vae.compile(optimizer=optimizers.Adam(learning_rate=lr))

    def _prepare_vae_inputs(self, X):
        X = np.asarray(X)
        if X.ndim == 2:
            X = X[..., np.newaxis]
        X = X.astype(np.float32)
        return (X, X[..., :self.M_output])

    def fit(self, channel, model_path: Optional[str] = None):
        callbacks = [keras.callbacks.EarlyStopping(min_delta=1e-3, patience=20, verbose=1, mode='min')]
        if model_path is not None:
            callbacks.append(keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=1, mode='min', save_best_only=True))
        
        train_inputs = self._prepare_vae_inputs(channel)
        self.history_ = self.vae.fit(train_inputs, epochs=self.epochs, validation_split=0.2, callbacks=callbacks, verbose=True)
        return self

    def alpha_selection(self, X, y, model_path: Optional[str] = None, load_model=False, custom_metrics=False):
        if model_path is not None and load_model:
            self.vae = keras.models.load_model(model_path, custom_objects={'sampling': Sampling}, compile=False)
        
        inp = Input(shape=(self.T, self.M))
        output = Input(shape=(self.T, self.M_output))
        x = self.vae([inp, output])
        out = Lambda(lambda y: [y[0][:, -1, :], y[1][:, -1, :]])(x)
        inference_model = Model([inp, output], out)

        if isinstance(X, list):
            X = np.asarray(X)
        batch = X[..., np.newaxis]
        prediction = inference_model([batch, batch[..., :self.M_output]])
        reconstruct = prediction[0].numpy()
        sig = np.sqrt(np.exp(prediction[1].numpy()))

        X_evaluate = X[:, self.T - 1:]
        y_evaluate = y
        best_f1 = np.zeros(self.M_output)
        max_alpha = 7
        self.alpha_up = max_alpha * np.ones(self.M_output)
        self.alpha_down = max_alpha * np.ones(self.M_output)

        for alpha_up in np.arange(max_alpha, 1, -1):
            for alpha_down in np.arange(max_alpha, 1, -1):
                pre_predict = ((X_evaluate < reconstruct - alpha_down * sig) | (X_evaluate > reconstruct + alpha_up * sig)).astype(int)
                for c in range(self.M_output):
                    f1_value = per_event_f_score(y_evaluate, pre_predict[:, c], beta=0.5 if custom_metrics else 1.0)
                    if f1_value >= best_f1[c]:
                        best_f1[c] = f1_value
                        self.alpha_up[c] = alpha_up
                        self.alpha_down[c] = alpha_down
        return self

    def predict(self, channel, model_path: Optional[str] = None, load_model=False, load_alpha=True, alpha_set_up=[], alpha_set_down=[]):
        if model_path is not None and load_model:
            self.vae = keras.models.load_model(model_path, custom_objects={'sampling': Sampling}, compile=False)
        
        inp = Input(shape=(self.T, self.M))
        output = Input(shape=(self.T, self.M_output))
        x = self.vae([inp, output])
        out = Lambda(lambda y: [y[0][:, -1, :], y[1][:, -1, :]])(x)
        inference_model = Model([inp, output], out)

        if isinstance(channel, (np.ndarray, list)):
            X = np.asarray(channel)
            if X.ndim == 2: X = X[..., np.newaxis]
            X = X.astype(np.float32)
            reconstruct, log_var = inference_model.predict([X, X[..., :self.M_output]], verbose=0)
            sig = np.sqrt(np.exp(log_var))
            X_evaluate = X[:, -1, :self.M_output]
        else:
            raise ValueError("Unsupported data type for channel prediction")

        aup = np.array(alpha_set_up) if len(alpha_set_up) == self.M_output else self.alpha_up
        adown = np.array(alpha_set_down) if len(alpha_set_down) == self.M_output else self.alpha_down
        
        pred = (X_evaluate < reconstruct - adown*sig) | (X_evaluate > reconstruct + aup*sig)
        return pred.astype(int)

from spaceai.models.classifiers.base import BaseClassifier

class DCVAEClassifier(BaseClassifier):
    def __init__(self, *args, alpha: Optional[float] = None, callback_handler=None, scale_data: bool = True, **kwargs):
        super().__init__(callback_handler=callback_handler)
        self.model = DCVAE(*args, **kwargs)
        self.alpha = alpha
        self.scale_data = scale_data
        if self.scale_data:
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()

    def _scale_input(self, X: Union[np.ndarray, list], fit: bool = False) -> np.ndarray:
        X_arr = np.asarray(X)
        if not self.scale_data:
            return X_arr
        
        orig_shape = X_arr.shape
        # Reshape to 2D for scaler: (N * T, M) or (N, M)
        X_2d = X_arr.reshape(-1, orig_shape[-1])
        if fit:
            X_2d = self.scaler.fit_transform(X_2d)
        else:
            X_2d = self.scaler.transform(X_2d)
        return X_2d.reshape(orig_shape)

    def fit(self, X: Union[np.ndarray, list], y: Optional[np.ndarray] = None, results: Optional[Dict[str, Any]] = None, **kwargs) -> "DCVAEClassifier":
        with self._callback_context("classifier_fit", results):
            X_scaled = self._scale_input(X, fit=True)
            self.model.fit(X_scaled)
            if self.alpha is not None:
                self.model.alpha_up = np.array([self.alpha] * self.model.M_output)
                self.model.alpha_down = np.array([self.alpha] * self.model.M_output)
            else:
                self.model.alpha_selection(X_scaled, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[np.ndarray, list], results: Optional[Dict[str, Any]] = None, **kwargs) -> np.ndarray:
        with self._callback_context("classifier_predict", results):
            X_scaled = self._scale_input(X, fit=False)
            return self.model.predict(X_scaled)
