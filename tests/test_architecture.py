import pytest

import torch

import deep_volatility_models.architecture as architecture


BATCH_SIZE = 5
NOISE_DIM = 77
EMBEDDING_SYMBOLS = 9

logsoftmax = torch.nn.LogSoftmax(dim=1)


def test_min_max_clamping():
    clamper = architecture.MinMaxClamping()
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
    filter = architecture.MinMaxClamping()
    filter.train(False)
    x = MAGNITUDE * torch.randn(3, 2)
    y = filter(x)
    assert (y == x).all()


def test_gaussian_noise():
    SIGMA = 0.1
    noise = architecture.GaussianNoise(SIGMA)
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


# TODO - Add test cases for risk_neutral=True
@pytest.mark.parametrize(
    "batch_size, window_size, input_symbols, output_symbols, feature_dim,"
    "mixture_components, exogenous_dim, extra_mixing_layers,"
    "use_batch_norm, expect_value_error",
    [
        (13, 0, 1, None, 3, 5, 7, 2, True, False),  # Window size of zero
        (13, 4, 1, None, 3, 5, 7, 2, True, False),  # Chnage window size to 4
        (13, 16, 1, None, 3, 5, 7, 2, True, False),  # Chnage window size to 16
        (13, 64, 1, None, 3, 5, 7, 2, True, False),  # Change window size to 64
        (13, 256, 1, None, 3, 5, 7, 2, True, False),  # Change window size to 256
        (13, 64, 1, None, 3, 5, 0, 2, True, False),  # Without an exogenous input
        (13, 64, 1, None, 3, 5, 7, 0, True, False),  # Without extra mixing layers
        (13, 64, 13, None, 3, 5, 7, 2, True, False),  # Symbol dimension other than 1
        (13, 64, 13, 13, 3, 5, 7, 2, True, False),  # Speciying output symbol dim
        (13, 64, 13, 11, 3, 5, 7, 2, True, False),  # Differing input/output symbol dim
        (13, 64, 1, None, 3, 5, 7, 2, False, False),  # Without batch norm
        (13, 60, 13, 13, 3, 5, 7, 2, True, True),  # Window size is not valid
        (13, 0, 1, None, 3, 5, 0, 2, True, True),  # No Window AND no exogenous input
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
    extra_mixing_layers,
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

    This code actually tests three things:
    1) Does for the forward() method of the mixture network provide sane outputs
    2) Does the forward_unpacked() method of the mixture netowrk provide sane
    outputs
    3) Does the forward() method of the ModelAndEmbedding work after combining
    a mixture model with an embedding.

    """
    if expect_value_error:
        with pytest.raises(ValueError):
            mixture_model = architecture.MixtureModel(
                window_size,
                input_symbols,
                output_symbols,
                exogenous_dimension=exogenous_dim,
                output_head_factory=architecture.MultivariateMixtureHead,
                feature_dimension=feature_dim,
                mixture_components=mixture_components,
                extra_mixing_layers=extra_mixing_layers,
                use_batch_norm=use_batch_norm,
                risk_neutral=False,
            )
    else:
        # This is the base mixture model we're testing.
        mixture_model = architecture.MixtureModel(
            window_size,
            input_symbols,
            output_symbols,
            exogenous_dimension=exogenous_dim,
            output_head_factory=architecture.MultivariateMixtureHead,
            feature_dimension=feature_dim,
            mixture_components=mixture_components,
            extra_mixing_layers=extra_mixing_layers,
            use_batch_norm=use_batch_norm,
            risk_neutral=False,
        )
        # Also create an embedding to test that ModelWithEmbedding returns sane results
        embedding = torch.nn.Embedding(EMBEDDING_SYMBOLS, exogenous_dim)

        # Combing mixture_model with embedding in embedding_model
        embedding_model = architecture.ModelWithEmbedding(mixture_model, embedding)

        # Create some test inputs.
        # 1) time series data:
        ts_data = torch.randn((batch_size, input_symbols, window_size))
        # 2) exogenous data (here that comes from an embedding, but that's not
        # necessarily the case).)
        exogenous_data = (
            torch.randn(batch_size, exogenous_dim) if exogenous_dim > 0 else None
        )
        # 3) an encoding vecto to tes with embedding_model
        encoding = torch.randint(0, EMBEDDING_SYMBOLS, (batch_size,))

        # Below we call the forward() methods of mixture_model and
        # embedding_model and also the forward_unpacked() method of
        # mixture_model and make sure they return tensors with the correct dimensions.

        for train in (True, False):
            mixture_model.train(train)
            embedding_model.train(train)

            if output_symbols is None:
                output_symbols = input_symbols

            # Call forward_unpacked()
            log_p_u, mu_u, sigma_inv_u, latents_u = mixture_model.forward_unpacked(
                ts_data,
                exogenous_data,
            )

            # Call mixture_model.forward() with different variations
            if exogenous_data is None:
                log_p, mu, sigma_inv, latents = mixture_model(ts_data)
            else:
                log_p, mu, sigma_inv, latents = mixture_model(
                    (ts_data, exogenous_data),
                )

            # Call embedding_model.forward()
            log_p_e, mu_e, sigma_inv_e, latents_e = embedding_model((ts_data, encoding))

            assert log_p_u.shape == (batch_size, mixture_components)
            assert mu_u.shape == (batch_size, mixture_components, output_symbols)
            assert sigma_inv_u.shape == (
                batch_size,
                mixture_components,
                output_symbols,
                input_symbols,
            )
            assert latents_u.shape == (batch_size, feature_dim)

            assert log_p.shape == log_p_u.shape
            assert mu.shape == mu_u.shape
            assert sigma_inv.shape == sigma_inv_u.shape
            assert latents.shape == latents_u.shape

            assert log_p.shape == log_p_e.shape
            assert mu.shape == mu_e.shape
            assert sigma_inv.shape == sigma_inv_e.shape
            assert latents.shape == latents_e.shape

            # Confirm that the window_size property returns the correct size:
            assert window_size == mixture_model.window_size


# TODO - Add test cases for risk_neutral=True
@pytest.mark.parametrize(
    "batch_size, window_size, input_symbols, feature_dim,"
    "exogenous_dim, extra_mixing_layers,"
    "use_batch_norm, expect_value_error",
    [
        (13, 0, 1, 3, 7, 2, True, False),  # Window size of zero
        (13, 4, 1, 3, 7, 2, True, False),  # Chnage window size to 4
        (13, 16, 1, 3, 7, 2, True, False),  # Chnage window size to 16
        (13, 64, 1, 3, 7, 2, True, False),  # Change window size to 64
        (13, 256, 1, 3, 7, 2, True, False),  # Change window size to 256
        (13, 64, 1, 3, 0, 2, True, False),  # Without an exogenous input
        (13, 64, 1, 3, 7, 0, True, False),  # Without extra mixing layers
        (13, 64, 1, 3, 7, 2, False, False),  # Without batch norm
        (13, 60, 1, 3, 7, 2, True, True),  # Window size is not valid
        (13, 0, 1, 3, 0, 2, True, True),  # No Window AND no exogenous input
    ],
)
def test_basic_model(  # basic model referes to a non-mixture model
    batch_size,
    window_size,
    input_symbols,
    feature_dim,
    exogenous_dim,
    extra_mixing_layers,
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

    This code actually tests three things:
    1) Does for the forward() method of the mixture network provide sane outputs
    2) Does the forward_unpacked() method of the mixture netowrk provide sane
    outputs
    3) Does the forward() method of the ModelAndEmbedding work after combining
    a mixture model with an embedding.

    """
    if expect_value_error:
        with pytest.raises(ValueError):
            model = architecture.UnivariateModel(
                window_size,
                input_symbols,
                exogenous_dimension=exogenous_dim,
                feature_dimension=feature_dim,
                extra_mixing_layers=extra_mixing_layers,
                use_batch_norm=use_batch_norm,
                risk_neutral=False,
            )
    else:
        # This is the base model we're testing.
        model = architecture.UnivariateModel(
            window_size,
            input_symbols,
            exogenous_dimension=exogenous_dim,
            feature_dimension=feature_dim,
            extra_mixing_layers=extra_mixing_layers,
            use_batch_norm=use_batch_norm,
            risk_neutral=False,
        )
        # Also create an embedding to test that ModelWithEmbedding returns sane results
        embedding = torch.nn.Embedding(EMBEDDING_SYMBOLS, exogenous_dim)

        # Combing model with embedding in embedding_model
        embedding_model = architecture.ModelWithEmbedding(model, embedding)

        # Create some test inputs.
        # 1) time series data:
        ts_data = torch.randn((batch_size, input_symbols, window_size))
        # 2) exogenous data (here that comes from an embedding, but that's not
        # necessarily the case).)
        exogenous_data = (
            torch.randn(batch_size, exogenous_dim) if exogenous_dim > 0 else None
        )
        # 3) an encoding vecto to tes with embedding_model
        encoding = torch.randint(0, EMBEDDING_SYMBOLS, (batch_size,))

        # Below we call the forward() methods of model and
        # embedding_model and also the forward_unpacked() method of
        # model and make sure they return tensors with the correct dimensions.

        for train in (True, False):
            model.train(train)
            embedding_model.train(train)

            # Call forward_unpacked()
            mu_u, sigma_inv_u, latents_u = model.forward_unpacked(
                ts_data,
                exogenous_data,
            )

            # Call model.forward() with different variations
            if exogenous_data is None:
                mu, sigma_inv, latents = model(ts_data)
            else:
                mu, sigma_inv, latents = model(
                    (ts_data, exogenous_data),
                )

            # Call embedding_model.forward()
            mu_e, sigma_inv_e, latents_e = embedding_model((ts_data, encoding))

            assert mu_u.shape == (batch_size, input_symbols)
            assert sigma_inv_u.shape == (
                batch_size,
                input_symbols,
                input_symbols,
            )
            assert latents_u.shape == (batch_size, feature_dim)

            assert mu.shape == mu_u.shape
            assert sigma_inv.shape == sigma_inv_u.shape
            assert latents.shape == latents_u.shape

            assert mu.shape == mu_e.shape
            assert sigma_inv.shape == sigma_inv_e.shape
            assert latents.shape == latents_e.shape

            # Confirm that the window_size property returns the correct size:
            assert window_size == model.window_size


@pytest.mark.parametrize(
    "head_class, batch_size, input_symbols, output_symbols, feature_dim,"
    "mixture_components, expect_value_error",
    [
        (architecture.MultivariateMixtureHead, 13, 3, None, 5, 7, False),
        (architecture.MultivariateMixtureHead, 13, 3, 3, 5, 7, False),
        (architecture.MultivariateMixtureHead, 13, 3, 2, 5, 7, False),
        (architecture.UnivariateMixtureHead, 13, 1, 1, 5, 7, False),
        (architecture.UnivariateMixtureHead, 13, 3, None, 5, 7, True),
        (architecture.UnivariateMixtureHead, 13, 3, 3, 5, 7, True),
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
