from typing import Any, Callable, List, Optional, Sequence
from functools import wraps
import jax


def rank_zero_only(fn: Callable) -> Callable:
    """Function that can be used as a decorator to enable a function/method being called only on global rank 0."""

    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[Any]:
        # if rank_zero_only.rank == 0:
        if jax.process_index() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn