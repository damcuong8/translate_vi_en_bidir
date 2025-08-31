import math


def lr_cosine_schedule(t: int, a_max: float, a_min: float, T_w: int, T_c: int):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        t (int): Iteration number to get learning rate for.
        a_max (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        a_min (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        T_w (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        T_c (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    # warm up
    if t < T_w:
        a_t = (t / T_w) * a_max
    # Cosine annealing
    elif T_w <= t <= T_c:
        a_t = a_min + 1/2 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (a_max - a_min)
    # Post-annealing
    else:
        a_t = a_min