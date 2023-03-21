import numpy as np
from nn import nn, preprocess
from sklearn.metrics import mean_squared_error


def test_single_forward():
    net = nn.NeuralNetwork(nn_arch=[{"input_dim": 3, "output_dim": 1, "activation": "relu"}],
                           lr=0.1,
                           seed=0,
                           batch_size=1,
                           epochs=1,
                           loss_function="mse")

    W_curr = np.array([[1, 2, 3]])
    b_curr = np.array([[1]])
    A_prev = np.array([[1, 2, 3]])

    A, Z = net._single_forward(W_curr, b_curr, A_prev, "relu")

    assert np.allclose(A, [15]), "A_curr is wrong"
    assert np.allclose(Z, [15]), "Z_curr is wrong"

def test_forward():
    # We can set up a really simple network and test it on a simple case
    net = nn.NeuralNetwork(nn_arch=[{"input_dim": 3, "output_dim": 1, "activation": "relu"}],
                           lr=0.1,
                           seed=0,
                           batch_size=1,
                           epochs=1,
                           loss_function="mse")

    print(net._param_dict)
    output, cache = net.forward(np.array([0, 0, 0]))

    assert(np.allclose(cache['A0'], np.array([0, 0, 0]))), "A0 stored wrong"
    assert (np.allclose(cache['A1'], np.array([0.22408932]))), "A1 stored wrong"
    assert (np.allclose(cache['Z1'], np.array([0.22408932]))), "Z1 stored wrong"
def test_single_backprop():
    # Hand calculated this - easy values
    layers = [{"input_dim": 3, "output_dim": 1, "activation": "relu"}]
    net = nn.NeuralNetwork(nn_arch=layers,
                           lr=0.1,
                           seed=42,
                           batch_size=1,
                           epochs=1,
                           loss_function="mse")

    W_curr = np.array([[1, 2, 3]])
    Z_curr = np.array([[1]])
    b_curr = np.array([[2]])
    A_prev = np.array([[1, 2, 3]])
    dA_curr = np.array([[1]])

    dA_prev, dw_curr, db_curr = net._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, "relu")
    assert np.allclose(dA_prev, np.array([1,2,3])), "dA_prev wrong"
    assert np.allclose(dw_curr, np.array([1, 2, 3])), "dw_curr wrong"
    assert np.allclose(db_curr, np.array([1])), "db_curr wrong"
def test_predict():
    # Testing this against self-calculated values, we can set the seed so that the values are the same every time
    layers = [{"input_dim": 3, "output_dim": 1, "activation": "relu"}]
    net = nn.NeuralNetwork(nn_arch=layers,
                           lr=0.1,
                           seed=42,
                           batch_size=1,
                           epochs=1,
                           loss_function="mse")
    print(net._param_dict)
    pred = net.predict(1)
    assert np.allclose(pred, np.array([[0.2019744 ],[0.13847656],[0.21707184]])), "Prediction calculated wrong"

def test_binary_cross_entropy():
    layers = [{"input_dim": 3, "output_dim": 1, "activation": "relu"}]
    net = nn.NeuralNetwork(nn_arch=layers,
                           lr=0.1,
                           seed=0,
                           batch_size=1,
                           epochs=1,
                           loss_function="mse")

    y = np.array([1, 1, 1, 1])
    y_hat = np.array([0.5, 0.5, 0.5, 0.5])
    bce = net._binary_cross_entropy(y, y_hat)
    assert np.allclose(bce, 0.6931471805599453), "BCE calculated wrong"
def test_binary_cross_entropy_backprop():
    # Testing against easy known calculation
    layers = [{"input_dim": 3, "output_dim": 1, "activation": "relu"}]
    net = nn.NeuralNetwork(nn_arch=layers,
                           lr=0.1,
                           seed=42,
                           batch_size=1,
                           epochs=1,
                           loss_function="mse")

    y = np.array([1, 1, 1, 1])
    y_hat = np.array([0.5, 0.5, 0.5, 0.5])
    bce_back = net._binary_cross_entropy_backprop(y, y_hat)
    assert np.allclose(bce_back, np.array([-0.5, -0.5, -0.5, -0.5])), "BCE backprop calculated wrong"
def test_mean_squared_error():
    # I'll test this against scipy's MSE
    layers = [{"input_dim": 3, "output_dim": 1, "activation": "relu"}]
    net = nn.NeuralNetwork(nn_arch=layers,
                           lr=0.1,
                           seed=42,
                           batch_size=1,
                           epochs=1,
                           loss_function="mse")

    y = np.array([1, 1, 1, 1])
    y_hat = np.array([0.5, 0.5, 0.5, 0.5])
    mse = net._mean_squared_error(y, y_hat)
    true_mse = mean_squared_error(y, y_hat)
    assert np.allclose(mse, true_mse)
def test_mean_squared_error_backprop():
    # I'm going to compare the net function to a known calculated value
    layers = [{"input_dim": 3, "output_dim": 1, "activation": "relu"}]
    net = nn.NeuralNetwork(nn_arch=layers,
                           lr=0.1,
                           seed=42,
                           batch_size=1,
                           epochs=1,
                           loss_function="mse")

    y = np.array([1, 1, 1, 1])
    y_hat = np.array([0.5, 0.5, 0.5, 0.5])
    mse_back = net._mean_squared_error_backprop(y, y_hat)
    assert np.allclose(mse_back, np.array([-1., -1., -1., -1.])), "MSE back prop calculated wrong"

def test_sample_seqs():
    seqs = ["A", "T", "C", "G", "Z"]
    labels = [True, True, True, True, False]
    expected_seqs = ["A", "T", "C", "G", "Z", "Z","Z","Z"]
    expected_labels = [True, True, True, True, False, False, False, False]
    ss, sl = preprocess.sample_seqs(seqs, labels)

    assert sorted(ss) == sorted(expected_seqs), "Wrong sampled sequences"
    assert sorted(expected_labels) == sorted(sl), "Wrong sampled labels"

def test_one_hot_encode_seqs():
    seqs = ["ATCG"]
    ec = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])
    seq_oh = preprocess.one_hot_encode_seqs(seqs)
    assert np.array_equal(ec, seq_oh), "One-hot encoding doesn't work as expected"
