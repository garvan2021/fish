
def eval_perplexity():
    return

def to_cpu(x):
    import numpy
    if x.__class__ == numpy.ndarray:
        return x
    return numpy.asnumpy(x)

def to_gpu(x):
    import cupy
    if x.__class__ == cupy.ndarray:
        return x
    return cupy.asarray(x)
