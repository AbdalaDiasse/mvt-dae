# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:31:12 2017

@author: XQing
"""

'''
MMD functions implemented in tensorflow.
'''
#from __future__ import division

import tensorflow as tf

from tf_ops import dot, sq_sum
#import utils 


_eps=1e-8

################################################################################
### Quadratic-time MMD with Gaussian RBF kernel

def _mix_rbf_kernel(X, Y, sigmas, wts=None):
    if wts is None:
        wts = [1] * len(sigmas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * tf.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * tf.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * tf.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)
    

def rbf_mmd2(X, Y, sigma=1, biased=True):
    return mix_rbf_mmd2(X, Y, sigmas=[sigma], biased=biased)


def mix_rbf_mmd2(X, Y, sigmas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


def rbf_mmd2_and_ratio(X, Y, sigma=1, biased=True):
    return mix_rbf_mmd2_and_ratio(X, Y, sigmas=[sigma], biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigmas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


###############################################################################
### Helper functions to compute variances based on kernel matrices


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
#    m = tf.cast(K_XX.get_shape()[0], tf.float32)
#    n = tf.cast(K_YY.get_shape()[0], tf.float32)
    m = 1500
    n = 800
    
    if biased:
          A = tf.reduce_sum(K_XX) / (m * m)
          B = tf.reduce_sum(K_YY) / (n * n)
          C = - 2 * tf.reduce_sum(K_XY) / (m * n)
          mmd2 = A + B+ C
#        mmd2 = (tf.reduce_sum(K_XX) / (m * m) + tf.reduce_sum(K_YY) / (n * n)
#              - 2 * tf.reduce_sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
              + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
              - 2 * tf.reduce_sum(K_XY) / (m * n))

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False,
                    min_var_est=_eps):
    mmd2, var_est = _mmd2_and_variance(
        K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    ratio = mmd2 / tf.sqrt(tf.maximum(var_est, min_var_est))
    return mmd2, ratio


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(K_XX.get_shape()[0], tf.float32)  # Assumes X, Y are same shape

    ### Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        const_diagonal = tf.cast(const_diagonal, tf.float32)
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = tf.diag_part(K_XX)
        diag_Y = tf.diag_part(K_YY)

        sum_diag_X = tf.reduce_sum(diag_X)
        sum_diag_Y = tf.reduce_sum(diag_Y)

        sum_diag2_X = sq_sum(diag_X)
        sum_diag2_Y = sq_sum(diag_Y)

    Kt_XX_sums = tf.reduce_sum(K_XX, 1) - diag_X
    Kt_YY_sums = tf.reduce_sum(K_YY, 1) - diag_Y
    K_XY_sums_0 = tf.reduce_sum(K_XY, 0)
    K_XY_sums_1 = tf.reduce_sum(K_XY, 1)

    Kt_XX_sum = tf.reduce_sum(Kt_XX_sums)
    Kt_YY_sum = tf.reduce_sum(Kt_YY_sums)
    K_XY_sum = tf.reduce_sum(K_XY_sums_0)

    Kt_XX_2_sum = sq_sum(K_XX) - sum_diag2_X
    Kt_YY_2_sum = sq_sum(K_YY) - sum_diag2_Y
    K_XY_2_sum  = sq_sum(K_XY)

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
              + (Kt_YY_sum + sum_diag_Y) / (m * m)
              - 2 * K_XY_sum / (m * m))
    else:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * (m-1))
              + (Kt_YY_sum + sum_diag_Y) / (m * (m-1))
              - 2 * K_XY_sum / (m * m))

    var_est = (
          2 / (m**2 * (m-1)**2) * (
              2 * sq_sum(Kt_XX_sums) - Kt_XX_2_sum
            + 2 * sq_sum(Kt_YY_sums) - Kt_YY_2_sum)
        - (4*m-6) / (m**3 * (m-1)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4*(m-2) / (m**3 * (m-1)**2) * (
              sq_sum(K_XY_sums_1) + sq_sum(K_XY_sums_0))
        - 4 * (m-3) / (m**3 * (m-1)**2) * K_XY_2_sum
        - (8*m - 12) / (m**5 * (m-1)) * K_XY_sum**2
        + 8 / (m**3 * (m-1)) * (
              1/m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - dot(Kt_XX_sums, K_XY_sums_1)
            - dot(Kt_YY_sums, K_XY_sums_0))
        )

    return mmd2, var_est

#########################################Joint MMD ###########################
def join_mix_rbf_kernel(list_X, list_Y, sigmas,num_layer,Ns,Nt,wts = None):
      list_K_XX = []
      list_K_XY = []
      list_K_YY = []
      for X, Y in zip(list_X,list_Y):
             K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y,sigmas , wts)
             list_K_XX.append(K_XX)
             list_K_XY.append(K_XY)
             list_K_YY.append(K_YY)
      K_XX = tf.convert_to_tensor(list_K_XX)
      K_XY = tf.convert_to_tensor(list_K_XY)
      K_YY = tf.convert_to_tensor(list_K_YY)

      return K_XX , K_XY, K_YY,d

def join_mix_rbf_mmd2(list_X, list_Y,num_layer,Ns,Nt,sigmas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = join_mix_rbf_kernel(list_X, list_Y, sigmas,num_layer,Ns,Nt,wts)
    mmd2 ,A ,B,C = join_mmd2(K_XX, K_XY, K_YY, num_layer,Ns,Nt,const_diagonal=d, biased=biased)
    return mmd2 ,A ,B,C, K_XX, K_XY, K_YY


def join_mmd2(K_XX, K_XY, K_YY, num_layer,Ns,Nt,const_diagonal=False, biased=False):
#    m = tf.cast(K_XX.get_shape()[0], tf.float32)
#    n = tf.cast(K_YY.get_shape()[0], tf.float32)
    m = Ns
    n = Nt
    
    if biased:
          D = tf.reduce_prod(K_XX,axis = 0)
          E = tf.reduce_sum(tf.reduce_prod(K_XX,axis = 0),axis = 0)
          F = tf.reduce_sum(tf.reduce_sum(tf.reduce_prod(K_XX,axis = 0),axis = 0)) 
          
          A = tf.reduce_sum(tf.reduce_sum(tf.reduce_prod(K_XX,axis = 0),axis = 0)) / (m * m)
          B = tf.reduce_sum(tf.reduce_sum(tf.reduce_prod(K_XY,axis = 0),axis = 0)) / (n * n)
          C = - 2 * tf.reduce_sum(tf.reduce_sum(tf.reduce_prod(K_YY,axis = 0),axis = 0))  / (m * n)
          mmd2 = A + B+ C
#         mmd2 = (tf.reduce_sum(K_XX) / (m * m) + tf.reduce_sum(K_YY) / (n * n)
#              - 2 * tf.reduce_sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
              + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
              - 2 * tf.reduce_sum(K_XY) / (m * n))

    return mmd2 ,D ,E,F

#def maximum_mean_discrepancy(x, y, kernel=utils.gaussian_kernel_matrix):
#  r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
#  Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
#  the distributions of x and y. Here we use the kernel two sample estimate
#  using the empirical mean of the two distributions.
#  MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
#              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
#  where K = <\phi(x), \phi(y)>,
#    is the desired kernel function, in this case a radial basis kernel.
#  Args:
#      x: a tensor of shape [num_samples, num_features]
#      y: a tensor of shape [num_samples, num_features]
#      kernel: a function which computes the kernel in MMD. Defaults to the
#              GaussianKernelMatrix.
#  Returns:
#      a scalar denoting the squared maximum mean discrepancy loss.
#  """
#  with tf.name_scope('MaximumMeanDiscrepancy'):
#    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
#    cost = tf.reduce_mean(kernel(x, x))
#    cost += tf.reduce_mean(kernel(y, y))
#    cost -= 2 * tf.reduce_mean(kernel(x, y))
#
#    # We do not allow the loss to become negative.
#    cost = tf.where(cost > 0, cost, 0, name='value')
#    return cost