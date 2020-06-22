# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""A PPO Agent implementing the clipped probability ratios.

Please see details of the algorithm in (Schulman,2017):
https://arxiv.org/abs/1707.06347.

Disclaimer: We intend for this class to eventually fully replicate:
https://github.com/openai/baselines/tree/master/baselines/ppo2

Currently, this agent surpasses the paper performance for average returns on
Half-Cheetah when wider networks and higher learning rates are used. However,
some details from this class still differ from the paper implementation.
For example, we do not perform mini-batch learning and learning rate annealing
yet. We are in working progress to reproduce the paper implementation exactly.

PPO is a simplification of the TRPO algorithm, both of which add stability to
policy gradient RL, while allowing multiple updates per batch of on-policy data.

TRPO enforces a hard optimization constraint, but is a complex algorithm, which
often makes it harder to use in practice. PPO approximates the effect of TRPO
by using a soft constraint. There are two methods presented in the paper for
implementing the soft constraint: an adaptive KL loss penalty, and
limiting the objective value based on a clipped version of the policy importance
ratio. This agent implements the clipped version.

The importance ratio clipping is described in eq (7) of
https://arxiv.org/pdf/1707.06347.pdf
- To disable IR clipping, set the importance_ratio_clipping parameter to 0.0.

Note that the objective function chooses the lower value of the clipped and
unclipped objectives. Thus, if the importance ratio exceeds the clipped bounds,
then the optimizer will still not be incentivized to pass the bounds, as it is
only optimizing the minimum.

Advantage is computed using Generalized Advantage Estimation (GAE):
https://arxiv.org/abs/1506.02438
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

from tf_agents.agents.ppo import ppo_agent


@gin.configurable
class PPOClipAgent(ppo_agent.PPOAgent):
  """A PPO Agent implementing the clipped probability ratios."""

  def __init__(self,
               time_step_spec,
               action_spec,
               optimizer=None,
               actor_net=None,
               value_net=None,
               importance_ratio_clipping=0.0,
               lambda_value=0.95,
               discount_factor=0.99,
               entropy_regularization=0.0,
               policy_l2_reg=0.0,
               value_function_l2_reg=0.0,
               shared_vars_l2_reg=0.0,
               value_pred_loss_coef=0.5,
               num_epochs=25,
               use_gae=False,
               use_td_lambda_return=False,
               normalize_rewards=True,
               reward_norm_clipping=10.0,
               normalize_observations=True,
               log_prob_clipping=0.0,
               gradient_clipping=None,
               check_numerics=False,
               debug_summaries=False,
               summarize_grads_and_vars=False,
               train_step_counter=None,
               name='PPOClipAgent'):
    """Creates a PPO Agent implementing the clipped probability ratios.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      optimizer: Optimizer to use for the agent.
      actor_net: A function actor_net(observations, action_spec) that returns
        tensor of action distribution params for each observation. Takes nested
        observation and returns nested action.
      value_net: A function value_net(time_steps) that returns value tensor from
        neural net predictions for each observation. Takes nested observation
        and returns batch of value_preds.
      importance_ratio_clipping: Epsilon in clipped, surrogate PPO objective.
        For more detail, see explanation at the top of the doc.
      lambda_value: Lambda parameter for TD-lambda computation.
      discount_factor: Discount factor for return computation.
      entropy_regularization: Coefficient for entropy regularization loss term.
      policy_l2_reg: Coefficient for l2 regularization of unshared policy
        weights.
      value_function_l2_reg: Coefficient for l2 regularization of unshared value
       function weights.
      shared_vars_l2_reg: Coefficient for l2 regularization of weights shared
        between the policy and value functions.
      value_pred_loss_coef: Multiplier for value prediction loss to balance with
        policy gradient loss.
      num_epochs: Number of epochs for computing policy updates.
      use_gae: If True (default False), uses generalized advantage estimation
        for computing per-timestep advantage. Else, just subtracts value
        predictions from empirical return.
      use_td_lambda_return: If True (default False), uses td_lambda_return for
        training value function. (td_lambda_return = gae_advantage +
        value_predictions)
      normalize_rewards: If true, keeps moving variance of rewards and
        normalizes incoming rewards.
      reward_norm_clipping: Value above and below to clip normalized reward.
      normalize_observations: If true, keeps moving mean and variance of
        observations and normalizes incoming observations.
      log_prob_clipping: +/- value for clipping log probs to prevent inf / NaN
        values.  Default: no clipping.
      gradient_clipping: Norm length to clip gradients.  Default: no clipping.
      check_numerics: If true, adds tf.debugging.check_numerics to help find NaN
        / Inf values. For debugging only.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If true, gradient summaries will be written.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      name: The name of this agent. All variables in this module will fall under
        that name. Defaults to the class name.

    Raises:
      ValueError: If the actor_net is not a DistributionNetwork.
    """
    super(PPOClipAgent, self).__init__(
        time_step_spec,
        action_spec,
        optimizer,
        actor_net,
        value_net,
        importance_ratio_clipping,
        lambda_value,
        discount_factor,
        entropy_regularization,
        policy_l2_reg,
        value_function_l2_reg,
        shared_vars_l2_reg,
        value_pred_loss_coef,
        num_epochs,
        use_gae,
        use_td_lambda_return,
        normalize_rewards,
        reward_norm_clipping,
        normalize_observations,
        gradient_clipping=gradient_clipping,
        check_numerics=check_numerics,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter,
        name=name,
        # Skips parameters used for the adaptive KL loss penalty version of PPO.
        log_prob_clipping=0.0,
        kl_cutoff_factor=0.0,
        kl_cutoff_coef=0.0,
        initial_adaptive_kl_beta=0.0,
        adaptive_kl_target=0.0,
        adaptive_kl_tolerance=0.0)
