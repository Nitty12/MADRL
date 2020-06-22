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

"""Tests for tf_agents.bandits.environments.mushroom_environment_utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.environments import mushroom_environment_utilities


class MushroomEnvironmentUtilitiesTest(tf.test.TestCase):

  def testOneHot(self):
    data = np.array([[1, 2], [1, 3], [2, 2], [1, 1]], dtype=np.int32)
    encoded = mushroom_environment_utilities._one_hot(data)
    expected = [[1, 0, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 0]]
    np.testing.assert_array_equal(encoded, expected)


if __name__ == '__main__':
  tf.test.main()
