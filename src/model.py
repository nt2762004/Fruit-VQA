from __future__ import annotations

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    AdditiveAttention,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Embedding,
    GlobalAveragePooling2D,
    Input,
    LSTM,
    MaxPooling2D,
    TimeDistributed,
)
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam


def build_classifier_model(input_shape=(256, 256, 3), num_classes: int = 10) -> Model:
    inputs = Input(shape=input_shape, name="image_input")

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax", name="class_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="fruit_classifier_cnn")
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_regression_model(input_shape=(256, 256, 3)) -> Model:
    inputs = Input(shape=input_shape, name="image_input")

    x = Conv2D(32, (3, 3), activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="linear", name="count_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="fruit_regression_cnn")
    model.compile(optimizer=Adam(learning_rate=3e-4), loss=Huber(delta=0.1), metrics=["mae"])
    return model


def build_vqa_model(
    vocab_size: int,
    max_length: int,
    num_fruit_classes: int,
    embedding_dim: int = 256,
    lstm_units: int = 512,
    use_attention: bool = False,
) -> Model:
    encoder_inputs = Input(shape=(max_length,), name="encoder_inputs")
    fruit_classifier_output = Input(shape=(num_fruit_classes,), name="fruit_classifier_output")
    fruit_regression_output = Input(shape=(1,), name="fruit_regression_output")
    decoder_inputs = Input(shape=(max_length - 1,), name="decoder_inputs")

    regression_dense = Dense(64, activation="relu")(fruit_regression_output)
    image_features = Concatenate(name="image_features")([fruit_classifier_output, regression_dense])
    image_dense = Dense(lstm_units, activation="relu")(image_features)

    encoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
    if use_attention:
        encoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    else:
        encoder_lstm = LSTM(lstm_units, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

    merged_state_h = Dense(lstm_units, activation="tanh")(Concatenate()([state_h, image_dense]))
    merged_state_c = Dense(lstm_units, activation="tanh")(Concatenate()([state_c, image_dense]))

    decoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[merged_state_h, merged_state_c])

    if use_attention:
        attention_output = AdditiveAttention(name="attention")([decoder_outputs, encoder_outputs])
        decoder_outputs = Concatenate(axis=-1)([decoder_outputs, attention_output])
        outputs = TimeDistributed(Dense(vocab_size, activation="softmax"))(decoder_outputs)
        model_name = "seq2seq_with_attention"
    else:
        outputs = Dense(vocab_size, activation="softmax")(decoder_outputs)
        model_name = "seq2seq_no_attention"

    model = Model(
        inputs=[encoder_inputs, fruit_classifier_output, fruit_regression_output, decoder_inputs],
        outputs=outputs,
        name=model_name,
    )
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
