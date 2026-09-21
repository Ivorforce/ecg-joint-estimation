"""1-D discrete wavelet transform layers.

Taken without functional changes from tensorflow-wavelets 1.1.2
(https://github.com/Timorleiderman/tensorflow-wavelets), which is released
under the MIT License:

    Copyright (c) 2021 Timor

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Vendored so that the package does not depend on tensorflow-probability, which
tensorflow-wavelets requires but these two layers do not use.
"""
import keras
import pywt
import tensorflow as tf


class IDWT1D(keras.layers.Layer):
    """Inverse 1-D DWT. Input: [batch, length, 2] (approximation, detail)."""

    def __init__(self, wavelet_name="haar", **kwargs):
        super().__init__(**kwargs)
        wavelet = pywt.Wavelet(wavelet_name)
        self.wavelet_name = wavelet_name
        self.rec_len = wavelet.rec_len
        self.rec_lpf = tf.constant(wavelet.rec_lo[::-1], dtype=tf.float32)[:, tf.newaxis, tf.newaxis]
        self.rec_hpf = tf.constant(wavelet.rec_hi[::-1], dtype=tf.float32)[:, tf.newaxis, tf.newaxis]
        self.border_padd = "REFLECT"
        self.border_type = "VALID"

    def call(self, inputs):
        a_ds, d_ds = tf.split(inputs, num_or_size_splits=2, axis=-1)
        batch_size, length, channels = tf.unstack(tf.shape(a_ds))

        # Upsample (interleave with zeros)
        upsampled_length = length * 2
        a_up = tf.reshape(tf.stack([a_ds, tf.zeros_like(a_ds)], axis=2), (batch_size, upsampled_length, channels))
        d_up = tf.reshape(tf.stack([d_ds, tf.zeros_like(d_ds)], axis=2), (batch_size, upsampled_length, channels))

        pad_size = self.rec_len - 1
        a_up = tf.pad(a_up, [[0, 0], [pad_size, pad_size], [0, 0]], self.border_padd)
        d_up = tf.pad(d_up, [[0, 0], [pad_size, pad_size], [0, 0]], self.border_padd)

        rec_lpf = tf.tile(self.rec_lpf, [1, 1, channels])
        rec_hpf = tf.tile(self.rec_hpf, [1, 1, channels])
        a_rec = tf.nn.conv1d(a_up, rec_lpf, stride=1, padding=self.border_type)
        d_rec = tf.nn.conv1d(d_up, rec_hpf, stride=1, padding=self.border_type)

        return a_rec[:, pad_size - 1:-pad_size, :] + d_rec[:, pad_size - 1:-pad_size, :]
