# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
from audioop import add
from typing import Any
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from src.utils import autoargs


def get_data_scaler(centered: bool):
  """Data normalizer. Assume data are always in [0, 1]."""
  if centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(centered: bool):
  """Inverse data normalizer."""
  if centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
    image,
    size=(resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = tf.round(h * ratio, tf.int32)
  w = tf.round(w * ratio, tf.int32)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset(config, additional_dim=None, uniform_dequantization=False, evaluation=False):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    additional_dim: An integer or `None`. If present, add one additional dimension to the output data,
      which equals the number of steps jitted together.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
  if batch_size % jax.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                     f'the number of devices ({jax.device_count()})')

  per_device_batch_size = batch_size // jax.device_count()
  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.AUTOTUNE
  num_epochs = None if not evaluation else 1
  # Create additional data dimension when jitting multiple steps together
  if additional_dim is None:
    batch_dims = [jax.local_device_count(), per_device_batch_size]
  else:
    batch_dims = [jax.local_device_count(), additional_dim, per_device_batch_size]

  # Create dataset builders for each dataset.
  if config.data.dataset == 'CIFAR10':
    dataset_builder = tfds.builder('cifar10')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'MNIST':
    dataset_builder = tfds.builder('mnist')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'SVHN':
    dataset_builder = tfds.builder('svhn_cropped')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)


  elif config.data.dataset == 'CELEBA':
    dataset_builder = tfds.builder('celeb_a')
    train_split_name = 'train'
    eval_split_name = 'validation'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = central_crop(img, 140)
      img = resize_small(img, config.data.image_size)
      return img

  elif config.data.dataset == 'ImageNet':
    size = {
      32: '32x32',
      64: '64x64'
    }[config.data.image_size]
    dataset_builder = tfds.builder(f'downsampled_imagenet/{size}')
    train_split_name = 'train'
    eval_split_name = 'validation'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return img


  elif config.data.dataset in ('FFHQ', 'CelebAHQ', 'LSUN'):
    dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
    train_split_name = eval_split_name = 'train'

  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')

  # Customize preprocess functions for each dataset.
  if config.data.dataset in ('FFHQ', 'CelebAHQ', 'LSUN'):
    def preprocess_fn(d):
      sample = tf.io.parse_single_example(d, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
      data = tf.io.decode_raw(sample['data'], tf.uint8)
      data = tf.reshape(data, sample['shape'])
      data = tf.transpose(data, (1, 2, 0))
      img = tf.image.convert_image_dtype(data, tf.float32)
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
      return dict(image=img, label=None)

  else:
    def preprocess_fn(d):
      """Basic preprocessing function scales data to [0, 1) and randomly flips."""
      img = resize_op(d['image'])
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        # img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
        img = (tf.random.uniform((config.data.image_size, config.data.image_size,
                                  config.data.num_channels), dtype=tf.float32) + img * 255.) / 256.

      return dict(image=img, label=d.get('label', None))

  def create_dataset(dataset_builder, split, is_train):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.threading.private_threadpool_size = 16
    dataset_options.threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(
        split=split, shuffle_files=is_train, read_config=read_config)
    else:
      ds = dataset_builder.with_options(dataset_options)
    ds = ds.repeat(count=num_epochs)
    if is_train: 
      ds = ds.shuffle(shuffle_buffer_size)
    # ds.interleave()
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    for batch_size in reversed(batch_dims):
      ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  train_ds = create_dataset(dataset_builder, train_split_name, True)
  eval_ds = create_dataset(dataset_builder, eval_split_name, False)
  return train_ds, eval_ds, dataset_builder

# def get_mnist_datasets():
#   """Load MNIST train and test datasets into memory."""
#   ds_builder = tfds.builder('mnist')
#   ds_builder.download_and_prepare()
#   train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
#   test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
#   train_ds['image'] = jnp.float32(train_ds['image']) / 255.
#   test_ds['image'] = jnp.float32(test_ds['image']) / 255.
#   return train_ds, test_ds

DS_NAMES = ['cifar10', 
            'mnist', 
            'svhn_cropped', 
            # 'celeb_a', 
            # 'downsampled_imagenet/32x32', 'downsampled_imagenet/64x64',
            # 'FFHQ', 'CelebAHQ', 'LSUN'
            ]

class DataModule:
  @autoargs(exclude=('additional_dim',))
  def __init__(self, 
               ds_name: Any, 
               image_size: int, 
               batch_size: int, 
               random_flip: bool, 
               centered: bool, 
               num_channels: bool, 
               uniform_dequantization: bool = False, 
               additional_dim = None,
               multi_gpu = None) -> None:
    if self.ds_name in DS_NAMES:
      self.dataset_builder = tfds.builder(self.ds_name)
      self.train_split_name = 'train'
      self.eval_split_name = 'test'
    else:
      raise NotImplementedError(f'Dataset {self.ds_name} not yet supported.')
    
    self.shuffle_buffer_size = 10000
    self.prefetch_size = tf.data.AUTOTUNE
    self.per_device_batch_size = self.batch_size
    self.batch_dims = []
    if multi_gpu is not None and multi_gpu > 0:
      self.batch_dims.append(multi_gpu)
      self.per_device_batch_size = self.batch_size // multi_gpu
    if additional_dim is not None and additional_dim > 0:
      self.batch_dims.append(additional_dim)
      self.additional_dim = additional_dim
    else:
      self.additional_dim = None
    self.batch_dims.append(self.per_device_batch_size)

    self.scaler = get_data_scaler(centered)
    self.inv_scaler = get_data_inverse_scaler(centered)


  def _get_process_fn(self, uniform_dequantization=False, evaluation=False):
    if self.ds_name in ('FFHQ', 'CelebAHQ', 'LSUN'):
      def preprocess_fn(d):
        sample = tf.io.parse_single_example(d, features={
          'shape': tf.io.FixedLenFeature([3], tf.int64),
          'data': tf.io.FixedLenFeature([], tf.string)})
        data = tf.io.decode_raw(sample['data'], tf.uint8)
        data = tf.reshape(data, sample['shape'])
        data = tf.transpose(data, (1, 2, 0))
        img = tf.image.convert_image_dtype(data, tf.float32)
        if self.random_flip and not evaluation:
          img = tf.image.random_flip_left_right(img)
        if uniform_dequantization:
          img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
        return dict(image=img, label=None)
    else:
      def resize_op(img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [self.image_size, self.image_size], antialias=True)

      def preprocess_fn(d):
        """Basic preprocessing function scales data to [0, 1) and randomly flips."""
        img = resize_op(d['image'])
        if self.random_flip and not evaluation:
          img = tf.image.random_flip_left_right(img)
        if uniform_dequantization:
          # img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
          img = (tf.random.uniform((self.image_size, self.image_size,
                                    self.num_channels), dtype=tf.float32) + img * 255.) / 256.

        return dict(image=img, label=d.get('label', None))
    return preprocess_fn

  def _create_dataset(self, preprocess_fn, is_train, 
                      private_threadpool_size=48, 
                      max_intra_op_parallelism=1,
                      num_epochs=None):
    # Create additional data dimension when jitting multiple steps together
    split = self.train_split_name if is_train else self.eval_split_name
    # num_epochs = None if is_train else 1
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.threading.private_threadpool_size = private_threadpool_size
    dataset_options.threading.max_intra_op_parallelism = max_intra_op_parallelism
    read_config = tfds.ReadConfig(options=dataset_options)
    if isinstance(self.dataset_builder, tfds.core.DatasetBuilder):
      self.dataset_builder.download_and_prepare()
      ds = self.dataset_builder.as_dataset(
        split=split, shuffle_files=is_train, read_config=read_config)
    else:
      ds = self.dataset_builder.with_options(dataset_options)
    ds = ds.repeat(count=num_epochs)
    if is_train: 
      ds = ds.shuffle(self.shuffle_buffer_size)
    # ds.interleave()
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    for batch_size in reversed(self.batch_dims):
      ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(self.prefetch_size)

  def train_ds(self):
    preprocess_fn = self._get_process_fn(uniform_dequantization=self.uniform_dequantization, 
                                          evaluation=True)
    return self._create_dataset(preprocess_fn, is_train=True)

  def test_ds(self):
    preprocess_fn = self._get_process_fn(uniform_dequantization=self.uniform_dequantization, 
                                          evaluation=False)
    return self._create_dataset(preprocess_fn, is_train=False)
    

def main():
  import time

  def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for i, sample in enumerate(dataset):
          # Performing a training step
          # time.sleep(0.01)
          if i >= 10000: break
    print("Execution time:", time.perf_counter() - start_time)
  

  dm = DataModule('mnist', 32, 16, False, False, False, 16, 5)
  train_ds = dm.train_ds()
  benchmark(train_ds, num_epochs=1)

if __name__ == '__main__':
  main()
  