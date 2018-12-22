from keras import activations, constraints, initializers, regularizers
# noinspection PyPep8Naming
import keras.backend as K
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.utils import conv_utils


class Masking2D(Layer):
    """
    Masks rows and columns of a 3D tensor along axes (1, 2) if all elements
    are equal to mask_value.
    """

    def __init__(self, mask_value=0., **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.mask_value = mask_value

    def compute_mask(self, inputs, mask=None):
        masked_cols = K.cast(K.any(K.not_equal(inputs, self.mask_value), axis=-1, keepdims=True), K.floatx())
        masked_rows = K.cast(K.any(K.not_equal(inputs, self.mask_value), axis=-2, keepdims=True), K.floatx())
        return K.batch_dot(masked_cols, masked_rows)

    def call(self, inputs, **kwargs):
        return inputs * self.compute_mask(inputs)

    def get_config(self):
        config = {'mask_value': self.mask_value}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape


class Softmax2D(Layer):
    """
    Layer that takes a 3D tensor with dimensions (batch_size, n_a, n_b)
    that is potentially masked (Masking or Masking2D) and applies a
    softmax transform along axis=2.
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

    def compute_mask(self, inputs, mask=None):
        # do not pass mask to next layers
        return None

    def call(self, inputs, mask=None, **kwargs):
        # column-wise max
        col_max = K.max(inputs, axis=-1, keepdims=True)

        # exponential
        a = K.exp(inputs - col_max)

        # apply mask after the exp; will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in Theano
            a *= K.cast(mask, K.floatx())

        # In some cases, especially in the early stages of training, the sum
        # may be almost zero and this results in NaNs. A workaround is to add
        # a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=-1, keepdims=True) + K.epsilon(), K.floatx())

        return a

    def compute_output_shape(self, input_shape):
        return input_shape


# noinspection PyUnusedLocal
class MaskedConv1D(Layer):
    """
    Masked 1D convolution with "same" padding.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 padding=None,
                 **kwargs):
        """

        :param filters: (int) dimensionality of the output space
                        (i.e. number of output filters in the convolution).
        :param kernel_size: (int, list[int], tuple[int]) dimensions of the convolution window.
        :param strides: (int, list[int], tuple[int]) strides of the convolution.
                        Specifying any `stride` value != 1 is incompatible with any
                        `dilation_rate` value != 1.
        :param data_format: (str) one of "channels_last" (default) or "channels_first".
                            The ordering of the dimensions in the inputs. "channels_last"
                            corresponds to inputs with shape (batch, ..., channels) while
                            "channels_first" corresponds to inputs with shape (batch, channels, ...).
                            This argument defaults to the `image_data_format` value found in
                            your Keras config file (~/.keras/keras.json). If you never set it
                            then it will be "channels_last".
        :param dilation_rate: (int, list[int], tuple[int]) the dilation rate(s) to use for
                              dilated convolution. Specifying any `dilation_rate` value != 1
                              is incompatible with any `strides` value != 1.
        :param activation: (str) activation fn to use. If you don't specify anything, no
                           activation is applied (i.e. "linear" activation: `a(x) = x`).
        :param use_bias: (bool) whether the layer uses a bias vector
        :param kernel_initializer: initializer for the `kernel` weights matrix
        :param bias_initializer: initializer for the bias vector
        :param kernel_regularizer: regularizer fn applied to the `kernel` weights matrix
        :param bias_regularizer: regularizer fn applied to the bias vector
        :param activity_regularizer: regularizer fn applied to the output of the layer (its "activation")
        :param kernel_constraint: constraint fn applied to the `kernel` weights matrix
        :param bias_constraint: constraint fn applied to the bias vector
        :param padding:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 1, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 1, 'strides')
        self.padding = conv_utils.normalize_padding('same')
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 1, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=3)
        self.supports_masking = True
        self.kernel = None
        self.bias = None
        self.built = False

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
        self.built = True

    def compute_mask(self, inputs, mask=None):
        # do not pass mask to next layers
        return None

    def call(self, inputs, mask=None, **kwargs):
        outputs = K.conv1d(inputs, self.kernel,
                           strides=self.strides[0],
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate[0])
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

        if mask is not None:
            outputs *= K.cast(K.expand_dims(mask), K.floatx())

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(space[i], self.kernel_size[i],
                                                        padding=self.padding,
                                                        stride=self.strides[i],
                                                        dilation=self.dilation_rate[i])
                new_space.append(new_dim)

            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        elif self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(space[i], self.kernel_size[i],
                                                        padding=self.padding,
                                                        stride=self.strides[i],
                                                        dilation=self.dilation_rate[i])
                new_space.append(new_dim)

            return (input_shape[0], self.filters) + tuple(new_space)
        else:
            raise ValueError('Invalid `data_format`. Expected one of: "channels_first", "channels_last".')

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return {**base_config, **config}


class RemoveMask(Layer):
    """
    Removes a mask in the sense that it does not pass the info to subsequent layers
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super().__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to next layers
        return None

    def call(self, inputs, **kwargs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


class _MaskedGlobalPool1D(Layer):
    """
    Abstract class for global pooling 1D layers
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def compute_mask(self, inputs, mask=None):
        # do not pass mask to next layers
        return None

    def call(self, inputs, **kwargs):
        raise NotImplementedError


class MaskedGlobalAveragePooling1D(_MaskedGlobalPool1D):
    """
    Global average pooling operation for temporal data
    """

    def call(self, inputs, mask=None, **kwargs):
        """

        :param inputs: 3D tensor with shape (batch_size, steps, features)
        :param mask:
        :param kwargs:
        :return: 2D tensor with shape (batch_size, features)
        """
        if mask is not None:
            s = K.sum(inputs, axis=1)
            c = K.sum(K.cast(K.expand_dims(mask), K.floatx()), axis=1)
            m = s / c
        else:
            m = K.mean(inputs, axis=1)

        return m


class MaskedGlobalMaxPooling1D(_MaskedGlobalPool1D):
    """
    Global average pooling operation for temporal data
    """

    def call(self, inputs, mask=None, **kwargs):
        """

        :param inputs: 3D tensor with shape (batch_size, steps, features)
        :param mask:
        :param kwargs:
        :return: 2D tensor with shape (batch_size, features)
        """
        return K.max(inputs, axis=1)


# Aliases
MaskedGlobalAvgPool1D = MaskedGlobalAveragePooling1D
MaskedGlobalMaxPool1D = MaskedGlobalMaxPooling1D
