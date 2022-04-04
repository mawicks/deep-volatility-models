import pytest

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


def test_untrained_mixmax_clamping_passes_all():
    MAGNITUDE = 1e6
    filter = network.MinMaxClamping()
    filter.train(False)
    x = MAGNITUDE * torch.randn(3, 2)
    y = filter(x)
    assert (y == x).all()


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


@pytest.mark.parametrize(
    "batch_size, window_size, input_symbols, output_symbols, feature_dim,"
    "mixture_components, exogenous_dim, use_batch_norm, expect_value_error",
    [
        (13, 0, 1, None, 3, 5, 7, True, False),  # Window size of zero
        (13, 4, 1, None, 3, 5, 7, True, False),  # Chnage window size to 4
        (13, 16, 1, None, 3, 5, 7, True, False),  # Chnage window size to 16
        (13, 64, 1, None, 3, 5, 7, True, False),  # Change window size to 64
        (13, 256, 1, None, 3, 5, 7, True, False),  # Change window size to 256
        (13, 64, 1, None, 3, 5, 0, True, False),  # Without an exogenous input
        (13, 64, 13, None, 3, 5, 7, True, False),  # Input symbol dimension other than 1
        (13, 64, 13, 13, 3, 5, 7, True, False),  # Speciying output symbol dim
        (13, 64, 13, 11, 3, 5, 7, True, False),  # Differing input/output symbol dim
        (13, 64, 1, None, 3, 5, 7, False, False),  # Without batch norm
        (13, 60, 13, 13, 3, 5, 7, True, True),  # Window size is not valid
        (
            13,
            0,
            1,
            None,
            3,
            5,
            0,
            True,
            True,
        ),  # Window size of zero AND no exogenous input
    ],
)
def test_mixture_model(
    batch_size,
    window_size,
    input_symbols,
    output_symbols,
    feature_dim,
    mixture_components,
    exogenous_dim,
    use_batch_norm,
    expect_value_error,
):
    """Test that a mmixture network can be created and evaluated
    with different internal feature dimensions.  This is only a sanity check
    that all of the dimensions conform and the network can produce output.
    These are untrained networks so that's all we expect.  There is more
    extensive validatation for unit tests of the individual head classes.  Here
    we also check that the network executes properly with the training flag on
    and off.

    """
    if expect_value_error:
        with pytest.raises(ValueError):
            g = network.MixtureModel(
                window_size,
                input_symbols,
                output_symbols,
                exogenous_dimension=exogenous_dim,
                output_head_type=network.MultivariateHead,
                feature_dimension=feature_dim,
                mixture_components=mixture_components,
                use_batch_norm=use_batch_norm,
            )
    else:
        g = network.MixtureModel(
            window_size,
            input_symbols,
            output_symbols,
            exogenous_dimension=exogenous_dim,
            output_head_type=network.MultivariateHead,
            feature_dimension=feature_dim,
            mixture_components=mixture_components,
            use_batch_norm=use_batch_norm,
        )
        for train in (True, False):
            g.train(train)
            log_p, mu, sigma_inv, latents = g(
                torch.randn((batch_size, input_symbols, window_size)),
                torch.randn(batch_size, exogenous_dim) if exogenous_dim > 0 else None,
            )

            assert log_p.shape == (batch_size, mixture_components)

            if output_symbols is None:
                output_symbols = input_symbols

            assert mu.shape == (batch_size, mixture_components, output_symbols)
            assert sigma_inv.shape == (
                batch_size,
                mixture_components,
                output_symbols,
                input_symbols,
            )

            assert latents.shape == (batch_size, feature_dim)

            # Confirm that the window_size property returns the correct size:
            assert window_size == g.window_size


@pytest.mark.parametrize(
    "head_class, batch_size, input_symbols, output_symbols, feature_dim,"
    "mixture_components, expect_value_error",
    [
        (network.MultivariateHead, 13, 3, None, 5, 7, False),
        (network.MultivariateHead, 13, 3, 3, 5, 7, False),
        (network.MultivariateHead, 13, 3, 2, 5, 7, False),
        (network.UnivariateHead, 13, 1, 1, 5, 7, False),
        (network.UnivariateHead, 13, 3, None, 5, 7, True),
        (network.UnivariateHead, 13, 3, 3, 5, 7, True),
    ],
)
def test_head_classes(
    head_class,
    batch_size,
    input_symbols,
    output_symbols,
    feature_dim,
    mixture_components,
    expect_value_error,
):
    """Test that a head network can be created and evaluated
    with different internal feature dimensions.  Also do some sanity checks on
    the output where the head should constrain it, such as having probabilities
    that add up to one and having an inverse sqrt of covariance matrix that is
    triangular. These are untrained networks so that's all we expect.  We also
    check that the network executes properly with the training flag on and off.
    """
    if expect_value_error:
        with pytest.raises(ValueError):
            head = head_class(
                input_symbols,
                output_symbols,
                feature_dimension=feature_dim,
                mixture_components=mixture_components,
            )
    else:
        head = head_class(
            input_symbols,
            output_symbols,
            feature_dimension=feature_dim,
            mixture_components=mixture_components,
        )
        for train in (True, False):
            head.train(train)
            log_p, mu, sigma_inv = head(torch.randn(batch_size, feature_dim))

            assert log_p.shape == (batch_size, mixture_components)

            # Make sure all probabilities add up to one
            # (logs add up to zero)
            assert torch.abs(torch.sum(torch.logsumexp(log_p, dim=1))) < 1e-5

            if output_symbols is None:
                output_symbols = input_symbols

            assert mu.shape == (batch_size, mixture_components, output_symbols)
            assert sigma_inv.shape == (
                batch_size,
                mixture_components,
                output_symbols,
                input_symbols,
            )

            is_lower_triangular(sigma_inv)
