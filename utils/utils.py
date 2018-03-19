
import tensorflow as tf
from tensorflow.python.client import timeline
import logging
import json

# https://github.com/ikhlestov/tensorflow_profiling/blob/master/03_merged_timeline_example.py
class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, step_stats):
        fetched_timeline = timeline.Timeline(step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()

        # convert chrome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)


def compute_time(start, end):
  seconds = end - start
  m, s = divmod(seconds, 60)
  h, m = divmod(m, 60)
  d, h = divmod(h, 24)
  print('Run time: %d:%02d:%02d:%02d' % (d, h, m, s))


def set_logger(log_path, mode='a'):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    # Logging to a file
    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


def show_variables(sess, variables):
  variables_names = [v.name for v in variables]
  values = sess.run(variables_names)
  for k, v in zip(variables_names, values):
      print('Variable: ', k)
      print('Shape: ', v.shape)


def add_trainable_variables_to_summary():
  for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)

