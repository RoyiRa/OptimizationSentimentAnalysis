# from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
# from tensorflow.python.ops import array_ops, control_flow_ops, math_ops, state_ops
# from tensorflow.python.framework import ops
#
#
# class Adam(OptimizerV2):
#     """  ### Creating a custom optimizer
#
#   If you intend to create your own optimization algorithm, simply inherit from
#   this class and override the following methods:
#
#     - `_resource_apply_dense` (update variable given gradient tensor is dense) - for everything else
#     - `_resource_apply_sparse` (update variable given gradient tensor is sparse) - for EMBEDDINGS
#     - `_create_slots` - use when defining tf.vars; weights' first and second order moments, use add_slot
#       (if your optimizer algorithm requires additional variables)
#     - `get_config`
#       (serialization of the optimizer, include all hyper parameters)
#   """
#
#
#     def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.999, name='MyAdam', **kwargs):
#         super(Adam, self).__init__(name, **kwargs)
#         self._set_hyper('learning_rate', kwargs.get('lr', alpha))
#         self._set_hyper('decay', self._initial_decay)
#         self._set_hyper('beta_1', beta_1)
#         self._set_hyper('beta_2', beta_2)
#
#     def _resource_apply_dense(self, grad, handle, apply_state=None):
#         """Add ops to apply dense gradients to the variable `handle`.
#
#         Args:
#           grad: a `Tensor` representing the gradient.
#           handle: a `Tensor` of dtype `resource` which points to the variable to be
#             updated.
#           apply_state: A dict which is used across multiple apply calls.
#
#         Returns:
#           An `Operation` which updates the value of the variable.
#         """
#         var_dtype = handle.dtype.base_dtype
#         lr_t = array_ops.identity(self._get_hyper('learning_rate', var_dtype))
#         beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
#         beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
#         epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)
#         m = self.get_slot(handle, 'm')
#         v = self.get_slot(handle, 'v')
#         local_step = math_ops.cast(self.iterations + 1, var_dtype)
#         next_step = math_ops.cast(self.iterations + 2, var_dtype)
#         decay_base = math_ops.cast(0.96, var_dtype)
#
#         # Learning rate multipliers
#         if self.lr_multipliers is not None:
#             lr_t = _apply_lr_multiplier(self, lr_t, handle)
#
#         # Due to the recommendations in [2], i.e. warming momentum schedule
#         momentum_cache_t = beta_1_t * (1. - 0.5 * (
#             math_ops.pow(decay_base, self._initial_decay * local_step)))
#         momentum_cache_t_1 = beta_1_t * (1. - 0.5 * (
#             math_ops.pow(decay_base, self._initial_decay * next_step)))
#         m_schedule_new = math_ops.cast(self._m_cache_read,
#                                        var_dtype) * momentum_cache_t
#         if var_dtype is self._m_cache.dtype:
#             m_schedule_new = array_ops.identity(state_ops.assign(
#                 self._m_cache, m_schedule_new, use_locking=self._use_locking))
#         m_schedule_next = m_schedule_new * momentum_cache_t_1
#
#         # the following equations given in [1]
#         g_prime = grad / (1. - m_schedule_new)
#         m_t = beta_1_t * m + (1. - beta_1_t) * grad
#         m_t_prime = m_t / (1. - m_schedule_next)
#         v_t = beta_2_t * v + (1. - beta_2_t) * math_ops.square(grad)
#         v_t_prime = v_t / (1. - math_ops.pow(beta_2_t, local_step))
#         m_t_bar = (1. - momentum_cache_t) * g_prime + (
#                 momentum_cache_t * m_t_prime)
#
#         m_t = state_ops.assign(m, m_t, use_locking=self._use_locking)
#         v_t = state_ops.assign(v, v_t, use_locking=self._use_locking)
#
#         var_t = math_ops.sub(var, self.eta_t * lr_t * m_t_bar / (
#                 math_ops.sqrt(v_t_prime + epsilon_t)))
#
#         # Weight decays
#         if handle.name in self.weight_decays.keys():
#             var_t = _apply_weight_decays(self, handle, var_t)
#
#         var_update = state_ops.assign(handle, var_t, use_locking=self._use_locking)
#
#         # Cosine annealing
#         (iteration_done, t_cur_update, eta_t_update
#          ) = _update_t_cur_eta_t_v2(self, lr_t, handle)
#         if iteration_done and not self._init_notified:
#             self._init_notified = True
#
#         updates = [var_update, m_t, v_t]
#         if iteration_done:
#             updates += [t_cur_update]
#         if self.use_cosine_annealing and iteration_done:
#             updates += [eta_t_update]
#         return control_flow_ops.group(*updates)
#
#     def _resource_apply_sparse(self, grad, handle, indices, apply_state):
#         """Add ops to apply sparse gradients to the variable `handle`.
#
#         Similar to `_apply_sparse`, the `indices` argument to this method has been
#         de-duplicated. Optimizers which deal correctly with non-unique indices may
#         instead override `_resource_apply_sparse_duplicate_indices` to avoid this
#         overhead.
#
#         Args:
#           grad: a `Tensor` representing the gradient for the affected indices.
#           handle: a `Tensor` of dtype `resource` which points to the variable to be
#             updated.
#           indices: a `Tensor` of integral type representing the indices for which
#             the gradient is nonzero. Indices are unique.
#           apply_state: A dict which is used across multiple apply calls.
#
#         Returns:
#           An `Operation` which updates the value of the variable.
#         """
#
#     def get_config(self):
#         config = super(Adam, self).get_config()
#         return config
#
#         # config.update({
#         #     'learning_rate': self._serialize_hyperparameter('learning_rate'),
#         #     'decay': self._serialize_hyperparameter('decay'),
#         #     'beta_1': self._serialize_hyperparameter('beta_1'),
#         #     'beta_2': self._serialize_hyperparameter('beta_2'),
#         #     'epsilon': self.epsilon,
#         #     'amsgrad': self.amsgrad,
#         #     'total_iterations': int(self.total_iterations),
#         #     'weight_decays': self.weight_decays,
#         #     'use_cosine_annealing': self.use_cosine_annealing,
#         #     'autorestart': self.autorestart,
#         #     't_cur': int(K_eval(self.t_cur)),
#         #     'eta_t': float(K_eval(self.eta_t)),
#         #     'eta_min': float(K_eval(self.eta_min)),
#         #     'eta_max': float(K_eval(self.eta_max)),
#         #     'init_verbose': self.init_verbose
#         # })
#         # return config
#
#     def _create_slots(self, var_list):
#         for var in var_list:
#             self.add_slot(var, 'm')
#         for var in var_list:
#             self.add_slot(var, 'v')
#         if self.amsgrad:
#             for var in var_list:
#                 self.add_slot(var, 'vhat')
#         self._updates_per_iter = len(var_list)
#
#
#
# #
# # from tensorflow.python.ops import control_flow_ops
# # from tensorflow.python.ops import math_ops
# # from tensorflow.python.ops import state_ops
# # from tensorflow.python.framework import ops
# # from tensorflow.python.training import optimizer
# # import tensorflow as tf
# # class PowerSign(OptimizerV2):
# #     """Implementation of PowerSign.
# #     See [Bello et. al., 2017](https://arxiv.org/abs/1709.07417)
# #     @@__init__
# #     """
# #
# #     def __init__(self, learning_rate=0.001, alpha=0.01, beta=0.5, use_locking=False, name="PowerSign"):
# #         super(PowerSign, self).__init__(use_locking, name)
# #         self._lr = learning_rate
# #         self._alpha = alpha
# #         self._beta = beta
# #
# #         # Tensor versions of the constructor arguments, created in _prepare().
# #         self._lr_t = None
# #         self._alpha_t = None
# #         self._beta_t = None
# #
# #     def _prepare(self, varlist):
# #         self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
# #         self._alpha_t = ops.convert_to_tensor(self._beta, name="alpha_t")
# #         self._beta_t = ops.convert_to_tensor(self._beta, name="beta_t")
# #
# #     def _create_slots(self, var_list):
# #         # Create slots for the first and second moments.
# #         for v in var_list:
# #             self._zeros_slot(v, "m", self._name)
# #
# #     def _apply_dense(self, grad, var):
# #         lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
# #         alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)
# #         beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)
# #
# #         eps = 1e-7  # cap for moving average
# #
# #         m = self.get_slot(var, "m")
# #         m_t = m.assign(tf.maximum(beta_t * m + eps, tf.abs(grad)))
# #
# #         var_update = state_ops.assign_sub(var, lr_t * grad * tf.exp(
# #             tf.math.log(alpha_t) * tf.sign(grad) * tf.sign(m_t)))  # Update 'ref' by subtracting 'value
# #         # Create an op that groups multiple operations.
# #         # When this op finishes, all ops in input have finished
# #         return control_flow_ops.group(*[var_update, m_t])
# #
# #     def _apply_sparse(self, grad, var):
# #         raise NotImplementedError("Sparse gradient updates are not supported.")
# #
# #     def get_config(self):
# #         raise NotImplementedError("get_config is not supported.")