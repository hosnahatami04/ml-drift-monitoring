"""Count-based windowing: predictions are grouped into fixed-size windows by
arrival order. Pure functions; the store provides the sequence numbers."""


def window_id_for(sequence: int, window_size: int) -> int:
    if sequence < 1:
        raise ValueError("sequence numbers start at 1")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    return (sequence - 1) // window_size


def is_window_complete(row_count: int, window_size: int) -> bool:
    return row_count >= window_size
