# import nest_asyncio
# nest_asyncio.apply()
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

np.random.seed(0)


def preprocess(dataset):

  def batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1, 784]),
        y=tf.reshape(element['label'], [-1, 1]))

  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


def make_federated_data(client_data, client_ids):
  return [
      preprocess(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]


def create_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(784,)),
      tf.keras.layers.Dense(10, kernel_initializer='zeros'),
      tf.keras.layers.Softmax(),
  ])


def model_fn():
  # TFF will call this within different graph contexts.
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=preprocessed_example_dataset.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


def create_mnist_variables():
  return MnistVariables(
      weights=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
          name='weights',
          trainable=True),
      bias=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(10)),
          name='bias',
          trainable=True),
      num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
      loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
      accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False))


def mnist_forward_pass(variables, batch):
  y = tf.nn.softmax(tf.matmul(batch['x'], variables.weights) + variables.bias)
  predictions = tf.cast(tf.argmax(y, 1), tf.int32)

  flat_labels = tf.reshape(batch['y'], [-1])
  loss = -tf.reduce_mean(
      tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1]))
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(predictions, flat_labels), tf.float32))

  num_examples = tf.cast(tf.size(batch['y']), tf.float32)

  variables.num_examples.assign_add(num_examples)
  variables.loss_sum.assign_add(loss * num_examples)
  variables.accuracy_sum.assign_add(accuracy * num_examples)

  return loss, predictions


def get_local_mnist_metrics(variables):
  return collections.OrderedDict(
      num_examples=variables.num_examples,
      loss=variables.loss_sum / variables.num_examples,
      accuracy=variables.accuracy_sum / variables.num_examples)


@tff.federated_computation
def aggregate_mnist_metrics_across_clients(metrics):
  return collections.OrderedDict(
      num_examples=tff.federated_sum(metrics.num_examples),
      loss=tff.federated_mean(metrics.loss, metrics.num_examples),
      accuracy=tff.federated_mean(metrics.accuracy, metrics.num_examples))


class MnistModel(tff.learning.Model):

  def __init__(self):
    self._variables = create_mnist_variables()

  @property
  def trainable_variables(self):
    return [self._variables.weights, self._variables.bias]

  @property
  def non_trainable_variables(self):
    return []

  @property
  def local_variables(self):
    return [
        self._variables.num_examples, self._variables.loss_sum,
        self._variables.accuracy_sum
    ]

  @property
  def input_spec(self):
    return collections.OrderedDict(
        x=tf.TensorSpec([None, 784], tf.float32),
        y=tf.TensorSpec([None, 1], tf.int32))

  @tf.function
  def forward_pass(self, batch, training=True):
    del training
    loss, predictions = mnist_forward_pass(self._variables, batch)
    num_exmaples = tf.shape(batch['x'])[0]
    return tff.learning.BatchOutput(
        loss=loss, predictions=predictions, num_examples=num_exmaples)

  @tf.function
  def report_local_outputs(self):
    return get_local_mnist_metrics(self._variables)

  @property
  def federated_output_computation(self):
    return aggregate_mnist_metrics_across_clients



if __name__ == '__main__':

    ### Prepare Input Data ###
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

    example_dataset = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[0])

    ### Preprocessing data ###
    NUM_CLIENTS = 10
    NUM_EPOCHS = 5
    BATCH_SIZE = 20
    SHUFFLE_BUFFER = 100
    PREFETCH_BUFFER = 10

    preprocessed_example_dataset = preprocess(example_dataset)

    sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

    federated_train_data = make_federated_data(emnist_train, sample_clients)

    ### Define variables ###
    MnistVariables = collections.namedtuple(
        'MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')

    optimizers = [tf.keras.optimizers.SGD, tf.keras.optimizers.Adam, tf.keras.optimizers.Adamax]

    best_score = 0
    best_optimizer = None
    best_client_lr = 0
    best_server_lr = 0

    # Find best Optimizer
    for optimizer in optimizers:

        for client_lr in np.arange(0, 0.05, 0.01).tolist():

            for server_lr in np.arange(0, 1, 0.01).tolist():

                ### Simulate Federated Training ###
                iterative_process = tff.learning.build_federated_averaging_process(
                    model_fn,
                    client_optimizer_fn=lambda: optimizer(learning_rate=client_lr),
                    server_optimizer_fn=lambda: optimizer(learning_rate=server_lr))

                state = iterative_process.initialize()

                for round_num in range(1, 12):
                    state, metrics = iterative_process.next(state, federated_train_data)

                ### Evaluate Performance ###
                evaluation = tff.learning.build_federated_evaluation(MnistModel)

                train_metrics = evaluation(state.model, federated_train_data)

                federated_test_data = make_federated_data(emnist_test, sample_clients)

                test_metrics = evaluation(state.model, federated_test_data)

                if best_score < (train_metrics['accuracy'] + test_metrics['accuracy']) / 2:

                    best_score = (train_metrics['accuracy'] + test_metrics['accuracy']) / 2
                    best_optimizer = optimizer
                    best_client_lr = client_lr
                    best_server_lr = server_lr


    print('*' * 10)
    print('Best Avg Accuracy: ', best_score)
    print('Best Optimizer: ', best_optimizer)
    print('Best Client LR: ', best_optimizer)
    print('Best Server LR: ', best_optimizer)