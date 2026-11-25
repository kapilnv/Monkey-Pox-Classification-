import numpy as np
from sklearn.metrics import accuracy_score

from SubFunctions import GetData
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import keras
from dbn import SupervisedDBNClassification



def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=keras.activations.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x


data = GetData.ReadDataset(dataset=2, exec=False).read_data()


x_train = data['Features']

x_train1 = np.expand_dims(x_train[:, :, :, 0], axis=-1)
x_train2 = np.expand_dims(x_train[:, :, :, 1], axis=-1)
x_train3 = np.expand_dims(x_train[:, :, :, 2], axis=-1)

y_train1 = data['Labels']



from keras.utils import to_categorical, plot_model
y_train = to_categorical(y_train1)

projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 2
chanDim = -1
num_classes = y_train.shape[1]

input_shape = (96, 96, 1)

print("Distributed learning-ensembled Multihead attentional DBN-ViT")

input_layer1 = Input(shape=input_shape)
input_layer2 = Input(shape=input_shape)
input_layer3 = Input(shape=input_shape)



x1 = Conv2D(32, (3, 3), padding="same")(input_layer1)
x1 = Activation("relu")(x1)
x1 = BatchNormalization(axis=chanDim)(x1)
x1 = MaxPooling2D(pool_size=(3, 3), strides=4)(x1)
x1 = Dropout(0.25)(x1)
x1 = Conv2D(64, (3, 3), padding="same")(x1)
x1 = Activation("relu")(x1)
x1 = BatchNormalization(axis=chanDim)(x1)
x1 = Conv2D(64, (3, 3), padding="same")(x1)
x1 = Activation("relu")(x1)
x1 = BatchNormalization(axis=chanDim)(x1)
x1 = MaxPooling2D(pool_size=(2, 2), strides=4)(x1)
x1 = Dropout(0.25)(x1)



x2 = Conv2D(32, (3, 3), padding="same")(input_layer2)
x2 = Activation("relu")(x2)
x2 = BatchNormalization(axis=chanDim)(x2)
x2 = MaxPooling2D(pool_size=(3, 3), strides=4)(x2)
x2 = Dropout(0.25)(x2)
x2 = Conv2D(64, (3, 3), padding="same")(x2)
x2 = Activation("relu")(x2)
x2 = BatchNormalization(axis=chanDim)(x2)
x2 = Conv2D(64, (3, 3), padding="same")(x2)
x2 = Activation("relu")(x2)
x2 = BatchNormalization(axis=chanDim)(x2)
x2 = MaxPooling2D(pool_size=(2, 2), strides=4)(x2)
x2 = Dropout(0.25)(x2)





x3 = Conv2D(32, (3, 3), padding="same")(input_layer3)
x3 = Activation("relu")(x3)
x3 = BatchNormalization(axis=chanDim)(x3)
x3 = MaxPooling2D(pool_size=(3, 3), strides=4)(x3)
x3 = Dropout(0.25)(x3)
x3 = Conv2D(64, (3, 3), padding="same")(x3)
x3 = Activation("relu")(x3)
x3 = BatchNormalization(axis=chanDim)(x3)
x3 = Conv2D(64, (3, 3), padding="same")(x3)
x3 = Activation("relu")(x3)
x3 = BatchNormalization(axis=chanDim)(x3)
x3 = MaxPooling2D(pool_size=(2, 2), strides=4)(x3)
x3 = Dropout(0.25)(x3)





x = MultiHeadAttention(
    num_heads=num_heads, key_dim=projection_dim, dropout=0.1
)(x1, x2, x3)

for _ in range(transformer_layers):
    # Layer normalization 1.
    x1 = LayerNormalization(epsilon=1e-6)(x)
    # Create a multi-head attention layer.
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=0.1
    )(x1, x1)
    # Skip connection 1.
    x2 = Add()([attention_output, x])
    # Layer normalization 2.
    x3 = LayerNormalization(epsilon=1e-6)(x2)
    # MLP.
    x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
    # Skip connection 2.
    x = Add()([x3, x2])

    # Fully connected layers
x = Flatten()(x)
x = Dense(1024)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(128)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(num_classes)(x)
output_layer = Activation("softmax")(x)
model = Model(inputs=[input_layer1, input_layer2, input_layer3], outputs=output_layer)
model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.fit([x_train1, x_train2, x_train3], y_train, epochs=100, batch_size=32, verbose=1, shuffle=True)

train_model = Model(inputs=model.input, outputs=model.layers[-11].output)
from keras.utils import plot_model

plot_model(train_model, to_file=f'Results\\Architecture.png', show_shapes=True, show_layer_names=True,
           show_layer_activations=True,
           show_dtype=True, dpi=1200)

train_features = train_model.predict(x_train)

dbn = SupervisedDBNClassification(
    hidden_layers_structure=[64, 32],  # DBN hidden layers
    learning_rate_rbm=0.01,
    learning_rate=0.1,
    n_epochs_rbm=10,
    n_iter_backprop=200,
    batch_size=16,
    activation_function='sigmoid',
    dropout_p=0.2
)
dbn.fit(train_features, y_train1)

yhat = np.array(dbn.predict(train_features))

print("Accuracy: %.2f%%" % (accuracy_score(y_train1, yhat) * 100))