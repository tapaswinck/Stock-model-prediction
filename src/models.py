"""
Model architectures for stock price sequence prediction.

Two architectures are provided:
  - build_lstm_model: the original 2-layer LSTM with dropout.
  - build_attention_model: an LSTM backbone with a self-attention layer over the
    timestep outputs, letting the model learn which days in the lookback window
    matter most for the prediction, rather than relying solely on the LSTM's
    final hidden state.

Neither architecture is claimed to be state-of-the-art for financial time series
— see the README's "Limitations" section for an honest discussion of what these
models can and cannot tell you.
"""

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input,
    Layer,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
)
import tensorflow as tf


def build_lstm_model(input_shape: tuple, lstm_units: int = 50, dropout: float = 0.2) -> Sequential:
    """Build the baseline 2-layer LSTM model.

    Args:
        input_shape: (timesteps, n_features).
        lstm_units: Number of units in each LSTM layer.
        dropout: Dropout rate applied after each LSTM layer.

    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=lstm_units, return_sequences=True),
        Dropout(dropout),
        LSTM(units=lstm_units),
        Dropout(dropout),
        Dense(units=1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def build_attention_model(
    input_shape: tuple,
    lstm_units: int = 64,
    num_heads: int = 4,
    key_dim: int = 16,
    dropout: float = 0.2,
) -> Model:
    """Build an LSTM + self-attention model.

    The LSTM processes the sequence and returns per-timestep outputs (rather than
    just the final hidden state). A multi-head self-attention layer then lets the
    model weigh different days in the lookback window against each other before
    pooling down to a single prediction.

    Args:
        input_shape: (timesteps, n_features).
        lstm_units: Number of units in the LSTM layer.
        num_heads: Number of attention heads.
        key_dim: Dimensionality of attention key/query projections per head.
        dropout: Dropout rate.

    Returns:
        Compiled Keras functional Model.
    """
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(units=lstm_units, return_sequences=True)(inputs)
    lstm_out = Dropout(dropout)(lstm_out)

    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(
        lstm_out, lstm_out
    )
    attn_out = LayerNormalization()(attn_out + lstm_out)  # residual connection

    pooled = GlobalAveragePooling1D()(attn_out)
    pooled = Dropout(dropout)(pooled)
    outputs = Dense(1)(pooled)

    model = Model(inputs, outputs, name="lstm_attention_model")
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def build_naive_baseline_predictions(y_true_scaled, lookback_value_idx=-1):
    """Naive 'no-skill' baseline: predict that tomorrow's price equals today's price.

    This is the single most important sanity check for any price-prediction model.
    If your model can't beat this trivial baseline on a real test set, it has not
    learned anything useful beyond the fact that prices are autocorrelated day to
    day (which is true of almost any asset and isn't a discovery).

    Args:
        y_true_scaled: The true (scaled) target sequence.
        lookback_value_idx: unused directly here; baseline is computed by the
            caller as y_true_scaled shifted by one step. Kept as a parameter for
            interface symmetry with model prediction functions.

    Returns:
        Naive predictions: y_true_scaled shifted forward by one day (first value
        repeated to keep array length aligned).
    """
    import numpy as np
    baseline = np.roll(y_true_scaled, 1)
    baseline[0] = y_true_scaled[0]
    return baseline
