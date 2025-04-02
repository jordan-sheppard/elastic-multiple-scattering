

class MaxIterationsExceedException(Exception):
    """An exception raised when an iterative algorithm has exceeded
    a given maximum number of iterations.
    """

class AlgorithmDivergedException(Exception):
    """An exception raised when an iterative algorithm has 
    an error that has diverged to an untolerably large number 
    (or to infinity or NaN).
    """