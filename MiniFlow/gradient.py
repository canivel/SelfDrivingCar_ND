def gradient_descent_update(x, gradx, learning_rate):
    """
    Performs a gradient descent update.
    """
    # TODO: Implement gradient descent.

    x = x - (gradx * learning_rate)
    return x
