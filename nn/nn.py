# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike


class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
            self,
            nn_arch: List[Dict[str, Union[int, str]]],
            lr: float,
            seed: int,
            batch_size: int,
            epochs: int,
            loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
            self,
            W_curr: ArrayLike,
            b_curr: ArrayLike,
            A_prev: ArrayLike,
            activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # First: linearly combine the previous layer and weights and add bias
        # This creates current linear transformed matrix

        Z_curr = np.dot(A_prev, W_curr.T) + b_curr.T

        # Employ activation function and then use that to activate the transformed matrix
        if activation == "sigmoid":
            A_curr = self._sigmoid(Z_curr)
        elif activation == "relu":
            A_curr = self._relu(Z_curr)
        elif activation == None:
            A_curr = Z_curr
        else:
            raise ValueError("Not acceptable activation function - activation functions accepted: relu and sigmoid")

        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        # Initialize cache dictionary:
        cache = {}
        cache['A0'] = X

        # Go through each layer:
        A_prev = X
        for idx, layer in enumerate(self.arch):
            # Parameter matrix has the parameters to use - layer index is 1+idx (see _init_params)
            layer_idx = idx + 1

            # Get the parameters and activation function for this layer
            activation = layer['activation']
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]

            # Do a single forward pass
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation)

            # Add A and Z matrices to the cache
            cache['A' + str(layer_idx)] = A_curr
            cache['Z' + str(layer_idx)] = Z_curr

            # Use current activation matrix for next layer
            A_prev = A_curr

        return A_curr, cache

    def _single_backprop(
            self,
            W_curr: ArrayLike,
            b_curr: ArrayLike,
            Z_curr: ArrayLike,
            A_prev: ArrayLike,
            dA_curr: ArrayLike,
            activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """

        # Find the right activation function for the back propagation:
        if activation_curr == "relu":
            activation_back = self._relu_backprop(dA_curr, Z_curr)
        elif activation_curr == "sigmoid":
            activation_back = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation_curr is None:
            activation_back = dA_curr
        else:
            raise ValueError("Backpropagation - Activation function not valid ")

        # Partial derivatives:
        dW_curr = np.dot(activation_back.T, A_prev)
        # NOT SURE ABOUT THIS: this db_curr is always the wrong shape
        db_curr = np.sum(activation_back, axis=0).reshape(b_curr.shape)
        dA_prev = np.dot(activation_back, W_curr)

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        # Initialize the dictionary
        grad_dict = {}

        # First backpropagation:
        if self._loss_func == "bce":
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)
        elif self._loss_func == "mse":
            dA_curr = self._mean_squared_error_backprop(y, y_hat)

        # Go through the layers - in reverse:
        for idx, layer in reversed(list(enumerate(self.arch))):
            activation = layer['activation']

            layer_idx = idx + 1
            W = self._param_dict["W" + str(layer_idx)]
            b = self._param_dict["b" + str(layer_idx)]

            Z_curr = cache['Z' + str(layer_idx)]
            A_prev = cache['A' + str(idx)]

            dA_prev, dW_curr, db_curr = self._single_backprop(W, b, Z_curr, A_prev, dA_curr, activation)

            dA_curr = dA_prev
            grad_dict['dw' + str(layer_idx)] = dW_curr
            grad_dict['db' + str(layer_idx)] = db_curr

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        # Go through each layer parameters, subtract the gradient scaled by learning rate
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            self._param_dict["W" + str(layer_idx)] -= self._lr * grad_dict["dw" + str(layer_idx)]
            self._param_dict["b" + str(layer_idx)] -= self._lr * grad_dict["db" + str(layer_idx)]

    def fit(
            self,
            X_train: ArrayLike,
            y_train: ArrayLike,
            X_val: ArrayLike,
            y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        # Initialize lists with per epoch training and validation loss
        per_epoch_loss_train = []
        per_epoch_loss_val = []
        for i in range(self._epochs):
            # Initialize training loss list
            training_loss = []

            # Create batches - just go through the samples and features and divide into the batch size
            batches = []
            batch = []

            # Shuffling the training data for each epoch of training
            shuffle = np.random.permutation(len(X_train))
            X_train_shuff = X_train[shuffle]
            y_train_shuff = y_train[shuffle]

            # Create batches
            #print(X_train.shape)
            num_batches = int(X_train.shape[0] / self._batch_size) + 1
            X_batch = np.array_split(X_train_shuff, num_batches)
            y_batch = np.array_split(y_train_shuff, num_batches)

            for X_train_batch, y_train_batch in zip(X_batch, y_batch):
                # Do a forward pass on the batch
                output, cache = self.forward(X_train_batch)

                # Append the loss
                if self._loss_func == "bce":
                    if y_train_batch.ndim == 1:
                        y_train_batch = y_train_batch.reshape(-1, 1)
                    loss_b = self._binary_cross_entropy(y_train_batch, output)
                    training_loss.append(loss_b)
                elif self._loss_func == "mse":
                    if y_train_batch.ndim == 1:
                        y_train_batch = y_train_batch.reshape(-1, 1)
                    loss_m = self._mean_squared_error(y_train_batch, output)
                    training_loss.append(loss_m)
                else:
                    raise ValueError("Loss function not valid")
                # Backprop and update parameters
                grad_dict = self.backprop(y_train_batch, output, cache)
                self._update_params(grad_dict)

            # Find average training loss and append to list
            avg_training_loss = np.mean(training_loss)
            per_epoch_loss_train.append(avg_training_loss)

            # Calculate validation loss
            val_predictions = self.predict(X_val)
            if self._loss_func == "bce":
                loss_b = self._binary_cross_entropy(y_val, val_predictions)
                per_epoch_loss_val.append(np.mean(loss_b))
            elif self._loss_func == "mse":
                loss_m = self._mean_squared_error(y_val, val_predictions)
                per_epoch_loss_val.append(np.mean(loss_m))
            else:
                raise ValueError("Loss function not valid")

        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, _ = self.forward(X)
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        There was a stack exchange post that helped me with this:
        https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth

        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        positive = Z >= 0
        negative = ~positive

        nl_transform = np.zeros_like(Z)
        nl_transform[positive] = 1 / (1 + np.exp(-Z[positive]))
        nl_transform[negative] = np.exp(Z[negative]) / (np.exp(Z[negative]) + 1)

        nl_transform = 1 / (1+np.exp(-Z))
        return nl_transform

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = dA * (self._sigmoid(Z) * (1 - self._sigmoid(Z)))
        return dZ

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = np.clip(Z, 0, None)
        return nl_transform

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = dA * (np.where(Z > 0, 1, 1e-5))

        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        loss = -(y * np.log(y_hat)) - ((1 - y) * np.log(1 - y_hat))
        loss = loss.mean()
        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        y_hat = np.clip(y_hat, 0.00001, 0.99999)
        dA = (((1 - y) / (1 - y_hat)) - (y / y_hat)) / len(y)
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        loss = np.square(y - y_hat)
        loss = loss.mean()
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        deriv = 2 * (y - y_hat)
        dA = -deriv
        return dA
