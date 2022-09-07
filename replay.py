# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Replay components for DQN-type agents."""

# pylint: disable=g-bad-import-order

import collections
import typing
from typing import Any, Callable, Generic, Iterable, Mapping, Optional, Sequence, Text, Tuple, TypeVar

import dm_env
import numpy as np
import snappy

from dqn_zoo import parts

CompressedArray = Tuple[bytes, Tuple, np.dtype]

# Generic replay structure: Any flat named tuple.
ReplayStructure = TypeVar('ReplayStructure', bound=Tuple[Any, ...])


class Transition(typing.NamedTuple):
  s_tm1: Optional[np.ndarray]
  a_tm1: Optional[parts.Action]
  r_t: Optional[float]
  discount_t: Optional[float]
  s_t: Optional[np.ndarray]


class UniformDistribution:
  """Provides uniform sampling of user-defined integer IDs."""

  def __init__(self, random_state: np.random.RandomState):
    self._random_state = random_state
    self._ids = []  # IDs in a contiguous indexable format for sampling.
    self._id_to_index = {}  # User ID -> index into self._ids.

  def add(self, ids: Sequence[int]) -> None:
    """Adds new IDs."""
    for i in ids:
      if i in self._id_to_index:
        raise IndexError('Cannot add ID %d, it already exists.' % i)

    for i in ids:
      idx = len(self._ids)
      self._id_to_index[i] = idx
      self._ids.append(i)

  def remove(self, ids: Sequence[int]) -> None:
    """Removes existing IDs."""
    for i in ids:
      if i not in self._id_to_index:
        raise IndexError('Cannot remove ID %d, it does not exist.' % i)

    for i in ids:
      idx = self._id_to_index[i]
      # Swap ID to be removed with ID at the end of self._ids.
      self._ids[idx], self._ids[-1] = self._ids[-1], self._ids[idx]
      self._id_to_index[self._ids[idx]] = idx  # Update index for swapped ID.
      self._id_to_index.pop(self._ids.pop())  # Remove ID from data structures.

  def sample(self, size: int) -> np.ndarray:
    """Returns sample of IDs, uniformly sampled."""
    indices = self._random_state.randint(self.size, size=size)
    ids = np.fromiter((self._ids[idx] for idx in indices),
                      dtype=np.int64,
                      count=len(indices))
    return ids

  def ids(self) -> Iterable[int]:
    """Returns an iterable of all current IDs."""
    return self._id_to_index.keys()

  @property
  def size(self) -> int:
    """Number of IDs currently tracked."""
    return len(self._ids)

  def get_state(self) -> Mapping[Text, Any]:
    """Retrieves distribution state as a dictionary (e.g. for serialization)."""
    return {
        'ids': self._ids,
        'id_to_index': self._id_to_index,
    }

  def set_state(self, state: Mapping[Text, Any]) -> None:
    """Sets distribution state from a (potentially de-serialized) dictionary."""
    self._ids = state['ids']
    self._id_to_index = state['id_to_index']

  def check_valid(self) -> Tuple[bool, str]:
    """Checks internal consistency."""
    if len(self._ids) != len(self._id_to_index):
      return False, 'ids and id_to_index should be the same size.'
    if len(self._ids) != len(set(self._ids)):
      return False, 'IDs should be unique.'
    if len(self._id_to_index.values()) != len(set(self._id_to_index.values())):
      return False, 'Indices should be unique.'
    for i in self._ids:
      if self._ids[self._id_to_index[i]] != i:
        return False, 'ID %d should map to itself.' % i
    # Indices map to themselves because of previous check and uniqueness.
    return True, ''


class TransitionReplay(Generic[ReplayStructure]):
  """Uniform replay, with LIFO storage for flat named tuples."""

  def __init__(self,
               capacity: int,
               structure: ReplayStructure,
               random_state: np.random.RandomState,
               encoder: Optional[Callable[[ReplayStructure], Any]] = None,
               decoder: Optional[Callable[[Any], ReplayStructure]] = None):
    self._capacity = capacity
    self._structure = structure
    self._random_state = random_state
    self._encoder = encoder or (lambda s: s)
    self._decoder = decoder or (lambda s: s)

    self._distribution = UniformDistribution(random_state=random_state)
    self._storage = collections.OrderedDict()  # ID -> item.
    self._t = 0  # Used to generate unique IDs for each item.

  def add(self, item: ReplayStructure) -> None:
    """Adds single item to replay."""
    if self.size == self._capacity:
      oldest_id, _ = self._storage.popitem(last=False)
      self._distribution.remove([oldest_id])

    item_id = self._t
    self._distribution.add([item_id])
    self._storage[item_id] = self._encoder(item)
    self._t += 1

  def get(self, ids: Sequence[int]) -> Iterable[ReplayStructure]:
    """Retrieves items by IDs."""
    for i in ids:
      yield self._decoder(self._storage[i])

  def sample(self, size: int) -> ReplayStructure:
    """Samples batch of items from replay uniformly, with replacement."""
    ids = self._distribution.sample(size)
    samples = self.get(ids)
    transposed = zip(*samples)
    stacked = [np.stack(xs, axis=0) for xs in transposed]
    return type(self._structure)(*stacked)  # pytype: disable=not-callable

  def ids(self) -> Iterable[int]:
    """Get IDs of stored transitions, for testing."""
    return self._storage.keys()

  @property
  def size(self) -> int:
    """Number of items currently contained in the replay."""
    return len(self._storage)

  @property
  def capacity(self) -> int:
    """Total capacity of replay (max number of items stored at any one time)."""
    return self._capacity

  def get_state(self) -> Mapping[Text, Any]:
    """Retrieves replay state as a dictionary (e.g. for serialization)."""
    return {
        # Serialize OrderedDict as a simpler, more common data structure.
        'storage': list(self._storage.items()),
        't': self._t,
        'distribution': self._distribution.get_state(),
    }

  def set_state(self, state: Mapping[Text, Any]) -> None:
    """Sets replay state from a (potentially de-serialized) dictionary."""
    self._storage = collections.OrderedDict(state['storage'])
    self._t = state['t']
    self._distribution.set_state(state['distribution'])

  def check_valid(self) -> Tuple[bool, str]:
    """Checks internal consistency."""
    if self._t < len(self._storage):
      return False, 't should be >= storage size.'
    if set(self._storage.keys()) != set(self._distribution.ids()):
      return False, 'IDs in storage and distribution do not match.'
    return self._distribution.check_valid()