import pytest
import numpy as np
import math
from .helpers import *

def multi_matrix_chain_rule_scalar(dfdgset,dgduset,num,vecvals):
    """
    find derivative of f(g1(u),...,gn(u)), where f is a matrix, gi are vectors or scalars, and u is a scalar. the indices of the gs that are vecs are indivated in vecvlas
    dfdgset = [dfdg1,...,dfdgn]. ith element is a list if gi is a vector. each list subelement is df/dgi[j] where gi[j] is the jth element of gi, etc. if gi is a scalar, ith element is df/dgi
    dgduset = [dg1du,...,dgndu]. ith element is a vector if gi is a vector and is dgi/du. if gi is a scalar, ith element is a scalar, dgi/du.
    """
    if len(dfdgset) != num:
        raise ValueError("wrong length of set in dfdg")
    if len(dgduset) != num:
        raise ValueError("wrong length of set in dgdu")
    if not np.all([isinstance(dfdgset[j],list) for j in vecvals]):
        raise ValueError("the elements in dfdgset with indices in vecvals should be lists")
    if not np.all([(isinstance(dfdgset[j],np.ndarray) or isinstance(dfdgset[j],np.matrix)) for j in range(num) if j not in vecvals]):
        raise ValueError("the elements in dfdgset with indices NOT in vecvals should be matrices (2-dimensional numpy arrays)")


    if not np.all([isinstance(dgduset[j],np.ndarray) for j in vecvals]):
        raise ValueError("the elements in dgduset with indices in vecvals should be vectors (2-dimensional numpy arrays, with second dim length == 1)")
    if not np.all([(isinstance(dgduset[j],float) or isinstance(dgduset[j],int) or (isinstance(dgduset[j],np.ndarray) and dgduset[j].size==1) ) for j in range(num) if j not in vecvals]):
        raise ValueError("the elements in dgduset with indices NOT in vecvals should be scalars")

    if 0 in vecvals:
        matrix_size_ref = dfdgset[0][0].shape
    else:
        matrix_size_ref = dfdgset[0].shape

    if not np.all([((isinstance(j,list) and np.all([k.shape == matrix_size_ref for k in j])) or (isinstance(j,np.ndarray) and j.shape == matrix_size_ref )) for j in dfdgset]):
        raise ValueError('not all in dfdgset have same output size')

    # print([isinstance(dgduset[j],np.ndarray) and dgduset[j].shape[1]==1 for j in vecvals])
    # print([isinstance(dgduset[j],float) or isinstance(dgduset[j],int) for j in range(num) if j not in vecvals])
    if not (np.all([isinstance(dgduset[j],np.ndarray) and dgduset[j].shape[1]==1 for j in vecvals]) and np.all([isinstance(dgduset[j],float) or isinstance(dgduset[j],int) for j in range(num) if j not in vecvals])):
        raise ValueError('not all in dgduset have scalar input length')

    if not np.all([len(dfdgset[j])==dgduset[j].shape[0] for j in vecvals]):
        raise ValueError('not all f input and g output have matching lengths (issue in vector components)')


    dfdgset2 = []
    dgduset2 = []
    for j in list(range(num)):
        if j in vecvals:
            dfdgset2 += dfdgset[j]
            dgduset2 += dgduset[j].flatten().tolist()
        else:
            dfdgset2 += [dfdgset[j]]
            dgduset2 += [dgduset[j]]
    return sum([dfdgset2[j]*dgduset2[j] for j in list(range(len(dfdgset2)))])#sum([np.squeeze(np.array(dfdgset[j]))@np.squeeze(np.array(dgduset[j])) for j in list(range(num))])


def multi_matrix_chain_rule_vector(dfdgset,dgduset,num,vecvals):
    """
    find derivative of f(g1(u),...,gn(u)), where f is a matrix, gi are vectors or scalars, and u is a vector. the indices of the gs that are vecs are indivated in vecvals
    dfdgset = [dfdg1,...,dfdgn]. ith element is a list if gi is a vector. each list subelement is df/dgi[j] where gi[j] is the jth element of gi, etc. if gi is a scalar, ith element is df/dgi
    dgduset = [dg1du,...,dgndu]. ith element is a matrix if gi is a vector and is dgi/du. if gi is a scalar, ith element is a vector, dgi/du.
    """
    if len(dfdgset) != num:
        raise ValueError("wrong length of set in dfdg")
    if len(dgduset) != num:
        raise ValueError("wrong length of set in dgdu")
    if not np.all([isinstance(dfdgset[j],list) for j in vecvals]):
        raise ValueError("the elements in dfdgset with indices in vecvals should be lists")
    if not np.all([(isinstance(dfdgset[j],np.ndarray) or isinstance(dfdgset[j],np.matrix)) for j in range(num) if j not in vecvals]):
        raise ValueError("the elements in dfdgset with indices NOT in vecvals should be matrices (2-dimensional numpy arrays)")


    if not np.all([(isinstance(dgduset[j],np.ndarray) or isinstance(dfdgset[j],np.matrix)) for j in vecvals]):
        raise ValueError("the elements in dgduset with indices in vecvals should be matrices (2-dimensional numpy arrays)")
    if not np.all([( (isinstance(dgduset[j],np.ndarray)or isinstance(dfdgset[j],np.matrix) ) and dgduset[j].shape[1]==1 ) for j in range(num) if j not in vecvals]):
        raise ValueError("the elements in dgduset with indices NOT in vecvals should be vectors(2-dimensional numpy arrays with second dim ==1)")

    if 0 in vecvals:
        matrix_size_ref = dfdgset[0][0].shape
        u_size_ref = dgduset[0].shape[1]
    else:
        matrix_size_ref = dfdgset[0].shape
        u_size_ref = dgduset[0].shape[0]

    if not np.all([((isinstance(j,list) and np.all([k.shape == matrix_size_ref for k in j])) or (isinstance(j,np.ndarray) and j.shape == matrix_size_ref )) for j in dfdgset]):
        raise ValueError('not all in dfdgset have same output size')

    # print([isinstance(dgduset[j],np.ndarray) and dgduset[j].shape[1]==1 for j in vecvals])
    # print([isinstance(dgduset[j],float) or isinstance(dgduset[j],int) for j in range(num) if j not in vecvals])
    # u_size_ref
    if not (np.all([ dgduset[j].shape[1]==u_size_ref for j in vecvals]) and np.all([dgduset[j].size == u_size_ref for j in range(num) if j not in vecvals])):
        raise ValueError('not all in dgduset have same input length')

    if not np.all([len(dfdgset[j])==dgduset[j].shape[0] for j in vecvals]):
        raise ValueError('not all f input and g output have matching lengths (issue in vector components)')


    dfdgset2 = []
    dgduset2 = []
    for j in list(range(num)):
        if j in vecvals:
            dfdgset2 += dfdgset[j]
            dgduset2 += [dgduset[j][k,:].reshape((1,dgduset[j].shape[1])) for k in range(dgduset[j].shape[0])]
        else:
            dfdgset2 += [dfdgset[j]]
            dgduset2 += [dgduset[j].T]
    return [sum([dfdgset2[j]*(dgduset2[j][:,k]).item() for j in list(range(len(dfdgset2)))]) for k in range(u_size_ref)]#sum([np.squeeze(np.array(dfdgset[j]))@np.squeeze(np.array(dgduset[j])) for j in list(range(num))])



def multi_vector_chain_rule(dfdgset,dgduset,num):
    """
    find derivative of f(g1(u),...,gn(u)), where f and gi and u are all vectors.
    dfdgset = [dfdg1,...,dfdgn]
    dgduset = [dg1du,...,dgndu]
    """
    if len(dfdgset) != num:
        raise ValueError("wrong length of set in dfdg")
    if len(dgduset) != num:
        raise ValueError("wrong length of set in dgdu")
    if not np.all([dfdgset[j].shape[0] == dfdgset[0].shape[0] for j in range(num)]):
        raise ValueError('not all in dfdgset have same output length')
    if not np.all([dgduset[j].shape[1] == dgduset[0].shape[1] for j in range(num)]):
        raise ValueError('not all in dgduset have same input length')
    if not np.all([dfdgset[j].shape[1] == dgduset[j].shape[0] for j in range(num)]):
        raise ValueError('not all f input and g output have matching lengths')

    return sum([np.squeeze(np.array(dfdgset[j]))@np.squeeze(np.array(dgduset[j])) for j in list(range(num))])
