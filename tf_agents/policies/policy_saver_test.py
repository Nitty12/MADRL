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

"""Tests for PolicySaver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.networks import actor_distribution_network
from tf_agents.networks import q_network
from tf_agents.networks import q_rnn_network
from tf_agents.policies import actor_policy
from tf_agents.policies import policy_saver
from tf_agents.policies import q_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.utils import test_utils


class PolicySaverTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super(PolicySaverTest, self).setUp()
    self._time_step_spec = ts.TimeStep(
        step_type=tensor_spec.BoundedTensorSpec(
            dtype=tf.int32, shape=(), name='st', minimum=0, maximum=2),
        reward=tensor_spec.BoundedTensorSpec(
            dtype=tf.float32, shape=(), name='reward', minimum=0.0,
            maximum=5.0),
        discount=tensor_spec.BoundedTensorSpec(
            dtype=tf.float32,
            shape=(),
            name='discount',
            minimum=0.0,
            maximum=1.0),
        observation=tensor_spec.BoundedTensorSpec(
            dtype=tf.float32,
            shape=(4,),
            name='obs',
            minimum=-10.0,
            maximum=10.0))
    self._action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=10, name='act_0')
    self._global_seed = 12345
    tf.compat.v1.set_random_seed(self._global_seed)

  def testUniqueSignatures(self):
    network = q_network.QNetwork(
        input_tensor_spec=self._time_step_spec.observation,
        action_spec=self._action_spec)

    policy = q_policy.QPolicy(
        time_step_spec=self._time_step_spec,
        action_spec=self._action_spec,
        q_network=network)

    saver = policy_saver.PolicySaver(policy, batch_size=None)
    action_signature_names = [
        s.name for s in saver._signatures['action'].input_signature
    ]
    self.assertAllEqual(
        ['0/step_type', '0/reward', '0/discount', '0/observation'],
        action_signature_names)
    initial_state_signature_names = [
        s.name for s in saver._signatures['get_initial_state'].input_signature
    ]
    self.assertAllEqual(['batch_size'], initial_state_signature_names)

  def testRenamedSignatures(self):
    time_step_spec = self._time_step_spec._replace(
        observation=tensor_spec.BoundedTensorSpec(
            dtype=tf.float32, shape=(4,), minimum=-10.0, maximum=10.0))

    network = q_network.QNetwork(
        input_tensor_spec=time_step_spec.observation,
        action_spec=self._action_spec)

    policy = q_policy.QPolicy(
        time_step_spec=time_step_spec,
        action_spec=self._action_spec,
        q_network=network)

    saver = policy_saver.PolicySaver(policy, batch_size=None)
    action_signature_names = [
        s.name for s in saver._signatures['action'].input_signature
    ]
    self.assertAllEqual(
        ['0/step_type', '0/reward', '0/discount', '0/observation'],
        action_signature_names)
    initial_state_signature_names = [
        s.name for s in saver._signatures['get_initial_state'].input_signature
    ]
    self.assertAllEqual(['batch_size'], initial_state_signature_names)

  def _convert_action_input_to_string_vector(self, action_input_tensors):
    action_input_tensors_strings = tf.nest.map_structure(
        tf.strings.as_string, action_input_tensors)

    return tf.concat([
        tf.expand_dims(action_input_tensors_strings[0].step_type, 1),
        tf.expand_dims(action_input_tensors_strings[0].reward, 1),
        tf.expand_dims(action_input_tensors_strings[0].discount, 1),
        action_input_tensors_strings[0].observation
    ], 1)

  def _convert_string_vector_to_action_input(self, example):
    return (ts.TimeStep(
        step_type=tf.cast(
            tf.strings.to_number(example[:, 0], tf.float32), tf.int32),
        reward=tf.strings.to_number(example[:, 1], tf.float32),
        discount=tf.strings.to_number(example[:, 2], tf.float32),
        observation=tf.strings.to_number(example[:, 3:7], tf.float32)), ())

  @parameterized.named_parameters(
      ('NotSeededNoStateNoInputFn', False, False, False, False),
      ('NotSeededWithStateNoInputFn', False, True, False, False),
      ('NotSeededDistributionNetworkNoInputFn', False, False, True, False),
      ('SeededNoStateNoInputFn', True, False, False, False),
      ('SeededWithStateNoInputFn', True, True, False, False),
      ('SeededDistributionNetworkNoInputFn', True, False, True, False),
      ('NotSeededNoStateInputFn', False, False, False, True),
      ('SeededNoStateInputFn', True, False, False, True),
  )
  def testSaveAction(self, seeded, has_state, distribution_net,
                     has_input_fn_and_spec):
    with tf.compat.v1.Graph().as_default():
      tf.compat.v1.set_random_seed(self._global_seed)
      with tf.compat.v1.Session().as_default():
        if distribution_net:
          network = actor_distribution_network.ActorDistributionNetwork(
              self._time_step_spec.observation, self._action_spec)
          policy = actor_policy.ActorPolicy(
              time_step_spec=self._time_step_spec,
              action_spec=self._action_spec,
              actor_network=network)
        else:
          if has_state:
            network = q_rnn_network.QRnnNetwork(
                input_tensor_spec=self._time_step_spec.observation,
                action_spec=self._action_spec)
          else:
            network = q_network.QNetwork(
                input_tensor_spec=self._time_step_spec.observation,
                action_spec=self._action_spec)

          policy = q_policy.QPolicy(
              time_step_spec=self._time_step_spec,
              action_spec=self._action_spec,
              q_network=network)

        action_seed = 98723

        batch_size = 3
        action_inputs = tensor_spec.sample_spec_nest(
            (self._time_step_spec, policy.policy_state_spec),
            outer_dims=(batch_size,),
            seed=4)
        action_input_values = self.evaluate(action_inputs)
        action_input_tensors = tf.nest.map_structure(tf.convert_to_tensor,
                                                     action_input_values)

        action_output = policy.action(*action_input_tensors, seed=action_seed)

        self.evaluate(tf.compat.v1.global_variables_initializer())

        action_output_dict = dict(((spec.name, value) for (spec, value) in zip(
            tf.nest.flatten(policy.policy_step_spec),
            tf.nest.flatten(action_output))))

        # Check output of the flattened signature call.
        (action_output_value, action_output_dict) = self.evaluate(
            (action_output, action_output_dict))

        input_fn_and_spec = None
        if has_input_fn_and_spec:
          input_fn_and_spec = (self._convert_string_vector_to_action_input,
                               tf.TensorSpec((7,), tf.string, name='example'))

        saver = policy_saver.PolicySaver(
            policy,
            batch_size=None,
            use_nest_path_signatures=False,
            seed=action_seed,
            input_fn_and_spec=input_fn_and_spec)
        path = os.path.join(self.get_temp_dir(), 'save_model_action')
        saver.save(path)

    with tf.compat.v1.Graph().as_default():
      tf.compat.v1.set_random_seed(self._global_seed)
      with tf.compat.v1.Session().as_default():
        reloaded = tf.compat.v2.saved_model.load(path)

        self.assertIn('action', reloaded.signatures)
        reloaded_action = reloaded.signatures['action']
        if has_input_fn_and_spec:
          self._compare_input_output_specs(
              reloaded_action,
              expected_input_specs=input_fn_and_spec[1],
              expected_output_spec=policy.policy_step_spec,
              batch_input=True)

        else:
          self._compare_input_output_specs(
              reloaded_action,
              expected_input_specs=(self._time_step_spec,
                                    policy.policy_state_spec),
              expected_output_spec=policy.policy_step_spec,
              batch_input=True)

        # Reload action_input_values as tensors in the new graph.
        action_input_tensors = tf.nest.map_structure(tf.convert_to_tensor,
                                                     action_input_values)

        action_input_spec = (self._time_step_spec, policy.policy_state_spec)
        function_action_input_dict = dict(
            (spec.name, value) for (spec, value) in zip(
                tf.nest.flatten(action_input_spec),
                tf.nest.flatten(action_input_tensors)))

        # NOTE(ebrevdo): The graph-level seeds for the policy and the reloaded
        # model are equal, which in addition to seeding the call to action() and
        # PolicySaver helps ensure equality of the output of action() in both
        # cases.
        self.assertEqual(reloaded_action.graph.seed, self._global_seed)

        def match_dtype_shape(x, y, msg=None):
          self.assertEqual(x.shape, y.shape, msg=msg)
          self.assertEqual(x.dtype, y.dtype, msg=msg)

        # The seed= argument for the SavedModel action call was given at
        # creation of the PolicySaver.
        if has_input_fn_and_spec:
          action_string_vector = self._convert_action_input_to_string_vector(
              action_input_tensors)
          reloaded_action_output_dict = reloaded_action(action_string_vector)
          reloaded_action_output = reloaded.action(action_string_vector)

        else:
          # This is the flat-signature function.
          reloaded_action_output_dict = reloaded_action(
              **function_action_input_dict)
          # This is the non-flat function.
          reloaded_action_output = reloaded.action(*action_input_tensors)

          if not has_state:
            # Try both cases: one with an empty policy_state and one with no
            # policy_state.  Compare them.

            # NOTE(ebrevdo): The first call to .action() must be stored in
            # reloaded_action_output because this is the version being compared
            # later against the true action_output and the values will change
            # after the first call due to randomness.
            reloaded_action_output_no_input_state = reloaded.action(
                action_input_tensors[0])
            # Even with a seed, multiple calls to action will get different
            # values, so here we just check the signature matches.
            tf.nest.map_structure(match_dtype_shape,
                                  reloaded_action_output_no_input_state,
                                  reloaded_action_output)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        (reloaded_action_output_dict,
         reloaded_action_output_value) = self.evaluate(
             (reloaded_action_output_dict, reloaded_action_output))

        self.assertAllEqual(action_output_dict.keys(),
                            reloaded_action_output_dict.keys())

        for k in action_output_dict:
          if seeded:
            self.assertAllClose(
                action_output_dict[k],
                reloaded_action_output_dict[k],
                msg='\nMismatched dict key: %s.' % k)
          else:
            match_dtype_shape(
                action_output_dict[k],
                reloaded_action_output_dict[k],
                msg='\nMismatch dict key: %s.' % k)

        # With non-signature functions, we can check that passing a seed does
        # the right thing the second time.
        if seeded:
          tf.nest.map_structure(self.assertAllClose, action_output_value,
                                reloaded_action_output_value)
        else:
          tf.nest.map_structure(match_dtype_shape, action_output_value,
                                reloaded_action_output_value)

  def testSaveGetInitialState(self):
    network = q_rnn_network.QRnnNetwork(
        input_tensor_spec=self._time_step_spec.observation,
        action_spec=self._action_spec)

    policy = q_policy.QPolicy(
        time_step_spec=self._time_step_spec,
        action_spec=self._action_spec,
        q_network=network)

    saver_nobatch = policy_saver.PolicySaver(
        policy, batch_size=None, use_nest_path_signatures=False)
    path = os.path.join(self.get_temp_dir(), 'save_model_initial_state_nobatch')
    saver_nobatch.save(path)
    reloaded_nobatch = tf.compat.v2.saved_model.load(path)
    self.assertIn('get_initial_state', reloaded_nobatch.signatures)
    reloaded_get_initial_state = (
        reloaded_nobatch.signatures['get_initial_state'])
    self._compare_input_output_specs(
        reloaded_get_initial_state,
        expected_input_specs=(tf.TensorSpec(
            dtype=tf.int32, shape=(), name='batch_size'),),
        expected_output_spec=policy.policy_state_spec,
        batch_input=False,
        batch_size=None)

    initial_state = policy.get_initial_state(batch_size=3)
    initial_state = self.evaluate(initial_state)

    reloaded_nobatch_initial_state = reloaded_nobatch.get_initial_state(
        batch_size=3)
    reloaded_nobatch_initial_state = self.evaluate(
        reloaded_nobatch_initial_state)
    tf.nest.map_structure(self.assertAllClose, initial_state,
                          reloaded_nobatch_initial_state)

    saver_batch = policy_saver.PolicySaver(
        policy, batch_size=3, use_nest_path_signatures=False)
    path = os.path.join(self.get_temp_dir(), 'save_model_initial_state_batch')
    saver_batch.save(path)
    reloaded_batch = tf.compat.v2.saved_model.load(path)
    self.assertIn('get_initial_state', reloaded_batch.signatures)
    reloaded_get_initial_state = reloaded_batch.signatures['get_initial_state']
    self._compare_input_output_specs(
        reloaded_get_initial_state,
        expected_input_specs=(),
        expected_output_spec=policy.policy_state_spec,
        batch_input=False,
        batch_size=3)

    reloaded_batch_initial_state = reloaded_batch.get_initial_state()
    reloaded_batch_initial_state = self.evaluate(reloaded_batch_initial_state)
    tf.nest.map_structure(self.assertAllClose, initial_state,
                          reloaded_batch_initial_state)

  def testNoSpecMissingOrColliding(self):
    spec_names = set()
    flat_spec = tf.nest.flatten(self._time_step_spec)
    missing_or_colliding = [
        policy_saver._true_if_missing_or_collision(s, spec_names)
        for s in flat_spec
    ]

    self.assertFalse(any(missing_or_colliding))

  def testTrueIfMissing(self):
    time_step_spec = self._time_step_spec._replace(
        observation=tensor_spec.BoundedTensorSpec(
            dtype=tf.float32, shape=(4,), minimum=-10.0, maximum=10.0))
    spec_names = set()
    flat_spec = tf.nest.flatten(time_step_spec)
    missing_or_colliding = [
        policy_saver._true_if_missing_or_collision(s, spec_names)
        for s in flat_spec
    ]

    self.assertTrue(any(missing_or_colliding))

  def testTrueIfCollision(self):
    time_step_spec = self._time_step_spec._replace(
        observation=tensor_spec.BoundedTensorSpec(
            dtype=tf.float32,
            shape=(4,),
            name='st',
            minimum=-10.0,
            maximum=10.0))
    spec_names = set()
    flat_spec = tf.nest.flatten(time_step_spec)
    missing_or_colliding = [
        policy_saver._true_if_missing_or_collision(s, spec_names)
        for s in flat_spec
    ]

    self.assertTrue(any(missing_or_colliding))

  def testRenameSpecWithNestPaths(self):
    time_step_spec = self._time_step_spec._replace(observation=[
        tensor_spec.TensorSpec(
            dtype=tf.float32,
            shape=(4,),
            name='obs1',
        ),
        tensor_spec.TensorSpec(
            dtype=tf.float32,
            shape=(4,),
            name='obs1',
        )
    ])

    renamed_spec = policy_saver._rename_spec_with_nest_paths(time_step_spec)

    new_names = [s.name for s in tf.nest.flatten(renamed_spec)]
    self.assertAllEqual(
        ['step_type', 'reward', 'discount', 'observation/0', 'observation/1'],
        new_names)

  def testTrainStepSaved(self):
    # We need to use one default session so that self.evaluate and the
    # SavedModel loader share the same session.
    with tf.compat.v1.Session().as_default():
      network = q_network.QNetwork(
          input_tensor_spec=self._time_step_spec.observation,
          action_spec=self._action_spec)

      policy = q_policy.QPolicy(
          time_step_spec=self._time_step_spec,
          action_spec=self._action_spec,
          q_network=network)
      self.evaluate(tf.compat.v1.initializers.variables(policy.variables()))

      train_step = common.create_variable('train_step', initial_value=7)
      self.evaluate(tf.compat.v1.initializers.variables([train_step]))

      saver = policy_saver.PolicySaver(
          policy, batch_size=None, train_step=train_step)
      path = os.path.join(self.get_temp_dir(), 'save_model')
      saver.save(path)

      reloaded = tf.compat.v2.saved_model.load(path)
      self.assertIn('get_train_step', reloaded.signatures)
      self.evaluate(tf.compat.v1.global_variables_initializer())
      train_step_value = self.evaluate(reloaded.train_step())
      self.assertEqual(7, train_step_value)
      train_step = train_step.assign_add(3)
      self.evaluate(train_step)
      saver.save(path)

      reloaded = tf.compat.v2.saved_model.load(path)
      self.evaluate(tf.compat.v1.global_variables_initializer())
      train_step_value = self.evaluate(reloaded.train_step())
      self.assertEqual(10, train_step_value)

  def testTrainStepNotSaved(self):
    network = q_network.QNetwork(
        input_tensor_spec=self._time_step_spec.observation,
        action_spec=self._action_spec)

    policy = q_policy.QPolicy(
        time_step_spec=self._time_step_spec,
        action_spec=self._action_spec,
        q_network=network)

    saver = policy_saver.PolicySaver(policy, batch_size=None)
    path = os.path.join(self.get_temp_dir(), 'save_model')

    saver.save(path)
    reloaded = tf.compat.v2.saved_model.load(path)

    self.assertIn('get_train_step', reloaded.signatures)
    train_step_value = self.evaluate(reloaded.train_step())
    self.assertEqual(-1, train_step_value)

  def testVariablesAccessible(self):
    network = q_network.QNetwork(
        input_tensor_spec=self._time_step_spec.observation,
        action_spec=self._action_spec)

    policy = q_policy.QPolicy(
        time_step_spec=self._time_step_spec,
        action_spec=self._action_spec,
        q_network=network)

    saver = policy_saver.PolicySaver(policy, batch_size=None)
    path = os.path.join(self.get_temp_dir(), 'save_model')

    self.evaluate(tf.compat.v1.global_variables_initializer())
    saver.save(path)
    reloaded = tf.compat.v2.saved_model.load(path)
    self.evaluate(
        tf.compat.v1.initializers.variables(reloaded.model_variables))

    model_variables = self.evaluate(policy.variables())
    reloaded_model_variables = self.evaluate(reloaded.model_variables)

    assert_np_all_equal = lambda a, b: self.assertTrue(np.equal(a, b).all())
    tf.nest.map_structure(assert_np_all_equal, model_variables,
                          reloaded_model_variables)

  def testCheckpointSave(self):
    network = q_network.QNetwork(
        input_tensor_spec=self._time_step_spec.observation,
        action_spec=self._action_spec)

    policy = q_policy.QPolicy(
        time_step_spec=self._time_step_spec,
        action_spec=self._action_spec,
        q_network=network)

    saver = policy_saver.PolicySaver(policy, batch_size=None)
    path = os.path.join(self.get_temp_dir(), 'save_model')

    self.evaluate(tf.compat.v1.global_variables_initializer())
    saver.save(path)
    checkpoint_path = os.path.join(self.get_temp_dir(), 'checkpoint')
    saver.save_checkpoint(checkpoint_path)

    self.assertTrue(tf.compat.v2.io.gfile.exists(checkpoint_path))

  def testUpdateWithCheckpoint(self):
    if not common.has_eager_been_enabled():
      self.skipTest('Only supported in TF2.x.')

    # Create and saved_model for a q_policy.
    network = q_network.QNetwork(
        input_tensor_spec=self._time_step_spec.observation,
        action_spec=self._action_spec)

    policy = q_policy.QPolicy(
        time_step_spec=self._time_step_spec,
        action_spec=self._action_spec,
        q_network=network)

    saver = policy_saver.PolicySaver(policy, batch_size=None)
    path = os.path.join(self.get_temp_dir(), 'save_model')

    self.evaluate(tf.compat.v1.global_variables_initializer())
    saver.save(path)

    # Assign -1 to all variables in the policy. Making checkpoint different than
    # the initial saved_model.
    self.evaluate(
        tf.nest.map_structure(lambda v: v.assign(v * 0 + -1),
                              policy.variables()))
    checkpoint_path = os.path.join(self.get_temp_dir(), 'checkpoint')
    saver.save_checkpoint(checkpoint_path)

    # Get an instance of the saved_model.
    reloaded_policy = tf.compat.v2.saved_model.load(path)
    self.evaluate(
        tf.compat.v1.initializers.variables(reloaded_policy.model_variables))

    # Verify loaded saved_model variables are different than the current policy.
    model_variables = self.evaluate(policy.variables())
    reloaded_model_variables = self.evaluate(reloaded_policy.model_variables)

    assert_np_not_equal = lambda a, b: self.assertFalse(np.equal(a, b).any())
    tf.nest.map_structure(assert_np_not_equal, model_variables,
                          reloaded_model_variables)

    # Update from checkpoint.
    checkpoint = tf.train.Checkpoint(policy=reloaded_policy)
    manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoint_path, max_to_keep=None)
    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    self.evaluate(
        tf.compat.v1.initializers.variables(reloaded_policy.model_variables))

    # Verify variables are now equal.
    model_variables = self.evaluate(policy.variables())
    reloaded_model_variables = self.evaluate(reloaded_policy.model_variables)

    assert_np_all_equal = lambda a, b: self.assertTrue(np.equal(a, b).all())
    tf.nest.map_structure(assert_np_all_equal, model_variables,
                          reloaded_model_variables)

  def testInferenceWithCheckpoint(self):
    if not common.has_eager_been_enabled():
      self.skipTest('Only supported in TF2.x.')

    # Create and saved_model for a q_policy.
    network = q_network.QNetwork(
        input_tensor_spec=self._time_step_spec.observation,
        action_spec=self._action_spec)

    policy = q_policy.QPolicy(
        time_step_spec=self._time_step_spec,
        action_spec=self._action_spec,
        q_network=network)
    sample_input = self.evaluate(
        tensor_spec.sample_spec_nest(self._time_step_spec, outer_dims=(3,)))

    saver = policy_saver.PolicySaver(policy, batch_size=None)
    path = os.path.join(self.get_temp_dir(), 'save_model')

    self.evaluate(tf.compat.v1.global_variables_initializer())
    original_eval = self.evaluate(policy.action(sample_input))
    saver.save(path)
    # Asign -1 to all variables in the policy. Making checkpoint different than
    # the initial saved_model.
    self.evaluate(
        tf.nest.map_structure(lambda v: v.assign(v * 0 + -1),
                              policy.variables()))
    checkpoint_path = os.path.join(self.get_temp_dir(), 'checkpoint')
    saver.save_checkpoint(checkpoint_path)

    # Get an instance of the saved_model.
    reloaded_policy = tf.compat.v2.saved_model.load(path)
    self.evaluate(
        tf.compat.v1.initializers.variables(reloaded_policy.model_variables))

    # Verify loaded saved_model variables are different than the current policy.
    model_variables = self.evaluate(policy.variables())
    reloaded_model_variables = self.evaluate(reloaded_policy.model_variables)

    assert_np_not_equal = lambda a, b: self.assertFalse(np.equal(a, b).any())
    tf.nest.map_structure(assert_np_not_equal, model_variables,
                          reloaded_model_variables)

    # Update from checkpoint.
    checkpoint = tf.train.Checkpoint(policy=reloaded_policy)
    manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoint_path, max_to_keep=None)
    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    self.evaluate(
        tf.compat.v1.initializers.variables(reloaded_policy.model_variables))

    # Verify variables are now equal.
    model_variables = self.evaluate(policy.variables())
    reloaded_model_variables = self.evaluate(reloaded_policy.model_variables)

    assert_np_all_equal = lambda a, b: self.assertTrue(np.equal(a, b).all())
    tf.nest.map_structure(assert_np_all_equal, model_variables,
                          reloaded_model_variables)

    # Verify variable update affects inference.
    reloaded_eval = self.evaluate(reloaded_policy.action(sample_input))
    tf.nest.map_structure(assert_np_not_equal, original_eval, reloaded_eval)
    current_eval = self.evaluate(policy.action(sample_input))
    tf.nest.map_structure(assert_np_not_equal, current_eval, reloaded_eval)

  def _compare_input_output_specs(self,
                                  function,
                                  expected_input_specs,
                                  expected_output_spec,
                                  batch_input,
                                  batch_size=None):
    args, kwargs = function.structured_input_signature
    self.assertFalse(args)

    def expected_spec(spec, include_batch_dimension):
      if include_batch_dimension:
        return tf.TensorSpec(
            dtype=spec.dtype,
            shape=tf.TensorShape([batch_size]).concatenate(spec.shape),
            name=spec.name)
      else:
        return spec

    expected_input_spec_dict = dict(
        (spec.name, expected_spec(spec, include_batch_dimension=batch_input))
        for spec in tf.nest.flatten(expected_input_specs))
    expected_output_spec_dict = dict(
        (spec.name, expected_spec(spec, include_batch_dimension=True))
        for spec in tf.nest.flatten(expected_output_spec))

    self.assertEqual(kwargs, expected_input_spec_dict)
    self.assertEqual(function.structured_outputs, expected_output_spec_dict)


if __name__ == '__main__':
  tf.test.main()
