__author__ = 'Daniel Neil'
import theano
import theano.tensor as T
from theano.scalar.basic import UnaryScalarOp, same_out_float_only, discrete_types
from theano.tensor.basic import _scal_elemwise, constructor
from theano import scalar as scal
from theano.tensor.elemwise import Elemwise
from theano.printing import pprint
import theano.printing as printing

class GradPreserveRoundOp(UnaryScalarOp):
    """
    Modified version of theano's rounding op

    Implement the same rounding algo as c round() fct.
    numpy.round fct IS DIFFERENT!
    See http://en.wikipedia.org/wiki/Rounding for more details.
    """
    def impl(self, x):
        return round_half_away_from_zero_vec(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        # Inserted here -- keep gradient!
        rval = gz
        #rval = x.zeros_like()

        if rval.type.dtype in discrete_types:
            rval = rval.astype(theano.config.floatX)

        return [rval]

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.outputs[0].type.dtype in ['float32', 'float64']:
            return "%(z)s = round(%(x)s);" % locals()
        else:
            Exception("The output should be float32 or float64")

def _spec_op_init(scalar_op, nfunc, nin, nout):
    def construct(symbol):
        symbolname = symbol.__name__
        msg = "no_inplace"
        n = "Elemwise{%s,%s}" % (symbolname, msg)
        rval = Elemwise(scalar_op, name=n,
            nfunc_spec=(nfunc and (nfunc, nin, nout)))

        if getattr(symbol, '__doc__', False):
            rval.__doc__ = symbol.__doc__ + '\n' + rval.__doc__

        # for the meaning of this see the ./epydoc script
        # it makes epydoc display rval as if it were a function, not an object
        rval.__epydoc_asRoutine = symbol
        rval.__module__ = 'tensor'

        pprint.assign(rval, printing.FunctionPrinter(symbolname))

        return rval
    return construct

grad_preserve_round = GradPreserveRoundOp(same_out_float_only)

@_spec_op_init(grad_preserve_round, None, None, None)
def grad_preserve_round_elemwise(a):
    """round_half_away_from_zero(a)"""

@constructor
def GradPreserveRoundTensor(a):
    """Rounding of a"""
    return grad_preserve_round_elemwise(a)
