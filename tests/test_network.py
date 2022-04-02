import torch

import network

BATCH_SIZE = 5
NOISE_DIM = 77

logsoftmax = torch.nn.LogSoftmax(dim=1)


def test_min_max_clamping():
    clamper = network.MinMaxClamping()
    x1 = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    x2 = torch.tensor([[4.0, 7.0], [5.0, 8.0], [6.0, 9.0]])
    # Column 1 ranges from 1 to 6
    # Column 2 ranges from 4 to 9
    # "Train" on x1 and x2
    clamper.train()
    assert clamper(x1) is x1
    assert clamper(x2) is x2

    # Evaluate on x_test
    clamper.eval()
    x_test = torch.tensor([[0.0, 3.0], [7.0, 10.0], [5.0, 5.0]])
    y = clamper(x_test)
    max_y = torch.max(y, dim=0)[0]
    min_y = torch.min(y, dim=0)[0]
    assert float(max_y[0]) <= 6.0
    assert float(max_y[1]) <= 9.0
    assert float(min_y[0]) >= 1.0
    assert float(min_y[1]) <= 4.0


def test_gaussian_noise():
    SIGMA = 0.1
    noise = network.GaussianNoise(SIGMA)
    x = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    y = noise(x)
    error = x - y
    squared_error = error * error
    rmse = torch.sqrt(torch.mean(squared_error))
    assert rmse > 0.1 * SIGMA
    assert rmse < 10.0 * SIGMA


def is_lower_triangular(m):
    mb, mixtures, oc, ic = m.shape
    col_offset = ic - oc
    assert ic >= oc

    for mb_i in range(mb):
        for mx_i in range(mixtures):
            for i in range(oc):
                for j in range(oc):
                    if j > i:
                        assert m[mb_i, mx_i, i, col_offset + j] == 0.0
                    else:
                        assert m[mb_i, mx_i, i, col_offset + j] != 0.0


def test_mixture_model():
    """Test that a mmixture network can be created and evaluated
    with different internal feature dimensions.  This is only a sanity
    check that all of the dimensions conform and the network can
    produce output.  These are untrained networks so that's all we
    expect.  We also check that the network executes properly with the
    training flag on and off.

    """

    for input_channels in range(3, 4):
        for output_channels in [None, 1, 2, input_channels]:
            for features in range(3, 5):
                for mixture in range(8, 10):
                    for window_size in [0, 16, 64, 256]:
                        for exogenous_dimension in [0, 4]:
                            if window_size == 0 and exogenous_dimension == 0:
                                continue

                            g = network.MixtureModel(
                                window_size,
                                input_channels,
                                output_channels,
                                exogenous_dimension=exogenous_dimension,
                                output_head_type=network.MultivariateHead,
                                feature_dimension=features,
                                mixture_components=mixture,
                            )

                            for train in (True, False):
                                g.train(train)
                                log_p, mu, sigma_inv, latents = g(
                                    torch.randn(
                                        (BATCH_SIZE, input_channels, window_size)
                                    ),
                                    torch.randn(BATCH_SIZE, exogenous_dimension)
                                    if exogenous_dimension > 0
                                    else None,
                                )

                                assert log_p.shape == (BATCH_SIZE, mixture)

                                # Make sure all probabilities add up to one
                                # (logs add up to zero)
                                assert (
                                    torch.abs(torch.sum(torch.logsumexp(log_p, dim=1)))
                                    < 1e-5
                                )

                                if output_channels is None:
                                    oc = input_channels
                                else:
                                    oc = output_channels

                                assert mu.shape == (BATCH_SIZE, mixture, oc)
                                assert sigma_inv.shape == (
                                    BATCH_SIZE,
                                    mixture,
                                    oc,
                                    input_channels,
                                )

                                is_lower_triangular(sigma_inv)

                                assert latents.shape == (BATCH_SIZE, features)
