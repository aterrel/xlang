from __future__ import absolute_import, division, print_function

import ctypes
from traceback import print_exc
import numpy
from dynd import nd, ndt, _lowlevel

class GraphT(ctypes.Structure):
    _fields_ = [('numNodes', ctypes.c_int),
                ('numEdges', ctypes.c_int),
                ('nodeNames', ctypes.POINTER(ctypes.c_char_p)),
                ('edgeValues', ctypes.POINTER(ctypes.c_double)),
                ('nodeValues', ctypes.POINTER(ctypes.c_double)),
                ('rowOffsets', ctypes.POINTER(ctypes.c_int)),
                ('colIndices', ctypes.POINTER(ctypes.c_int))]  

graphtpstr = """
{
 numNodes: int,
 numEdges: int,
 nodeNames: uintptr,
 edgeValues: pointer[%(numEdges)d * real],
 nodeValues: pointer[%(numNodes)d * real],
 rowOffsets: pointer[%(numRowOffsets)d * int],
 colIndices: pointer[%(numEdges)d * int]
}
"""

def print_graph(out_graph_addr, in_graph_addr):
    try:
        in_graph = GraphT.from_address(in_graph_addr)
        tp = ndt.type(graphtpstr % {'numNodes': in_graph.numNodes, 'numEdges': in_graph.numEdges, 'numRowOffsets': in_graph.numNodes + 1})
        print(tp)
        a = _lowlevel.array_from_ptr(tp, in_graph_addr, None, 'readonly')
        b = _lowlevel.array_from_ptr('(int, int, uintptr, uintptr, uintptr, uintptr, uintptr)', in_graph_addr, None, 'readonly')
        print('%r' % b)
        print('in_graph_addr ', hex(in_graph_addr))
        print('%r' % a.numNodes)
        print('%r' % a.numEdges)
        print('%r' % a.nodeNames)
        print('%r' % a.edgeValues)
        print('%r' % a.nodeValues)
        print('%r' % a.rowOffsets)
        print('%r' % a.colIndices)
        print(repr(a))
        """
        print(in_graph.numNodes)
        p = ctypes.cast(in_graph.nodeNames, ctypes.POINTER(ctypes.c_size_t))
        nodes = numpy.ctypeslib.as_array(p, shape=(in_graph.numNodes,))
        print(nodes, nodes.dtype)
        c = nodes.ctypes.data_as(ctypes.POINTER(ctypes.c_char_p))
        print(c[0], c[1])
        """
    except:
        print_exc()
        raise
