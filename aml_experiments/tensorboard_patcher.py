import os
import time
import warnings
from distutils.version import LooseVersion
from functools import wraps
from threading import local

_LOG_EVERY_N_STEPS = 1
THREAD_LOCAL = local()
LOG_CALLS_PER_MIN = int(os.environ.get("AMLT_MAX_AML_LOG_RATE", "200"))

# env var specifying allowlist containing metrics to be passed to AML
AMLT_TENSORBOARD_LOG_METRICS_ENV_VAR = "AMLT_TENSORBOARD_LOG_METRICS"


try:
  import ratelimit
  from wrapt import register_post_import_hook

  DEPENDENCIES_FOUND = True
except ImportError:
  warnings.warn(
    "Not patching tensorboard scalar logging for AML: Modules 'wrapt' and 'ratelimit' needed."
  )
  DEPENDENCIES_FOUND = False


class Flavor:
  pytorch = 1
  tensorflow = 2


def get_run_context():
  try:
    from azureml.core.run import Run

    run_context = Run.get_context()
  except ImportError:
    warnings.warn(
      "Cannot log to AML run: azureml could not be imported. "
      'Please "pip install azureml-defaults" in the container or image.'
    )
  except BaseException as exc:
    warnings.warn("Cannot log to AML run: could not obtain run context.")
    warnings.warn(str(exc))
  return run_context


if DEPENDENCIES_FOUND:

  class AMLRunLoggingWrapper:
    """
        Intercepts scalar logging function for tensorboard and sends the same info to the AML Run.

        It tries to handle situations where different patched implementations refer to each other.
        This is detected by tracking nested calls. If a nested call is detected, it is not logged to AML.
        This slightly convolutes the code, and currently pytorch/tf do not use each other's implementation,
        but we wanted to be future proof here.
        """

    def __init__(self):
      self.nested = False
      # metrics that need to be logged, if None, all metrics will be logged
      self.metrics_allowlist = None
      if AMLT_TENSORBOARD_LOG_METRICS_ENV_VAR in os.environ:
        self.metrics_allowlist = os.environ[AMLT_TENSORBOARD_LOG_METRICS_ENV_VAR].split(
          ","
        )

    @property
    def run_context(self):
      if not hasattr(self, "_run_context"):
        self._run_context = get_run_context()
      return self._run_context

    def log_to_run(self, metric, value):
      """
            Sends the metric + value to AML.

            If metrics_allowlist is set, only log if the metric is contained therein,
            otherwise all metrics all logged by default.
            """
      if not self.run_context:
        # AML not available, no need for further action
        return
      if self.metrics_allowlist is not None and metric not in self.metrics_allowlist:
        # This metric is not allowlisted, ignore
        return

      while True:
        try:
          self._log_to_run(metric, value)
          break
        except ratelimit.RateLimitException as exc:
          warnings.warn(
            "Logging metric '%s' to AML: cannot log faster than %s scalars per minute, stalling. "
            "Consider reducing log frequency, or setting the env variable %s (comma-separated) "
            "to specify which metrics to log, or set AMLT_NO_TENSORBOARD_PATCHING to disable "
            "automatic Tensorboard logging to AML completely."
            % (
              metric,
              LOG_CALLS_PER_MIN,
              AMLT_TENSORBOARD_LOG_METRICS_ENV_VAR,
            )
          )
          time.sleep(exc.period_remaining)

    @ratelimit.limits(calls=LOG_CALLS_PER_MIN, period=60)
    def _log_to_run(self, metric, value):
      self.run_context.log(metric, value)

    @staticmethod
    def _is_nested_context():
      if not hasattr(THREAD_LOCAL, "nested"):
        THREAD_LOCAL.nested = False
      return THREAD_LOCAL.nested

    def __call__(ctx, func, flavor):

      if flavor == Flavor.pytorch:

        def log_to_aml(args, kwargs):
          params = ctx.extract_args_pytorch(args, kwargs)
          if params:
            # some users may be logging scalar tensors...
            item_fn = getattr(params[1], "item", None)
            if callable(item_fn):
              params = [params[0], params[1].item()]
            ctx.log_to_run(*params)

      elif flavor == Flavor.tensorflow:

        def log_to_aml(args, kwargs):
          event = args[0] if args else kwargs["event"]
          # log_tf_event calls log_to_run after extracting scalars
          ctx.log_tf_event(event)

      @wraps(func)
      def inner(self, *args, **kwargs):
        if ctx._is_nested_context():
          return func(self, *args, **kwargs)

        old_nested_value = THREAD_LOCAL.nested

        THREAD_LOCAL.nested = True
        try:
          try:
            log_to_aml(args, kwargs)
          except BaseException:
            warnings.warn("Caught an error while logging to AML, ignoring.")
          return func(self, *args, **kwargs)
        finally:
          THREAD_LOCAL.nested = old_nested_value

      return inner

    @staticmethod
    def extract_args_pytorch(args, kwargs):
      """Extract tag name and scalar value from args/kwargs.

            Note that "self" here is e.g. the SummaryWriter object
            """
      if len(args) > 1:
        return args[0], args[1]
      if len(args) == 1 and "scalar_value" in kwargs:
        return args[0], kwargs["scalar_value"]
      if len(args) == 0 and "scalar_value" in kwargs and "tag" in kwargs:
        return kwargs["tag"], kwargs["scalar_value"]
      return None

    def log_tf_event(self, event):
      """
            Extracts metric information from the event protobuf.

            Taken from mlflow:
            https://github.com/mlflow/mlflow/blob/62d266f7cfbd41dc885b75705d4e08af84e368fe/mlflow/tensorflow.py
            """
      if event is None:
        return

      if event.WhichOneof("what") == "summary":
        summary = event.summary
        for v in summary.value:
          if v.HasField("simple_value"):
            # NB: Most TensorFlow APIs use one-indexing for epochs, while tf.Keras
            # uses zero-indexing. Accordingly, the modular arithmetic used here is slightly
            # different from the arithmetic used in `__MLflowTfKeras2Callback.on_epoch_end`,
            # which provides metric logging hooks for tf.Keras
            if (event.step - 1) % _LOG_EVERY_N_STEPS == 0:
              self.log_to_run(v.tag, v.simple_value)

  _aml_logging_wrapper = AMLRunLoggingWrapper()


def pytorch_imported(torch_tb):
  old_add_scalar = torch_tb.SummaryWriter.add_scalar
  torch_tb.SummaryWriter.add_scalar = wraps(old_add_scalar)(
    _aml_logging_wrapper(old_add_scalar, Flavor.pytorch)
  )


def tensorboard_imported(tb):
  try:
    # pretend that we cannot append to blobfuse files
    # so that pytorch writes to temp file and only overwrites
    # when flush() is called.
    from tensorboard.compat import tf

    try:
      delattr(tf.io.gfile.LocalFileSystem, "append")
    except AttributeError as ex:
      print("WARNING: not patching tensorboard for blobfuse: %s" % str(ex))
      pass
  except ImportError:
    pass


def tf_imported(tensorflow):
  """
    When tf is imported, we patch functions where events are added for Tensorboard.

    Here, we can't patch the scalar() function itself, since it may only return a node in a graph.
    The add_event function that we patch takes an protobuf "event", which needs to be parsed for
    scalar values that can be logged to AML.

    The code is loosely inspired by mlflow:
    https://github.com/mlflow/mlflow/blob/62d266f7cfbd41dc885b75705d4e08af84e368fe/mlflow/tensorflow.py
    """
  if LooseVersion(tensorflow.__version__) < LooseVersion("1.12"):
    warnings.warn(
      "Could not patch tensorflow to log to AML automatically. TensorFlow versions below 1.12 are not supported."
    )
    return

  try:
    from tensorflow.python.summary.writer.event_file_writer import EventFileWriter
    from tensorflow.python.summary.writer.event_file_writer_v2 import EventFileWriterV2
  except ImportError:
    warnings.warn(
      "Could not log to run. TensorFlow versions below 1.12 are not supported."
    )
    return

  to_patch = [
    (EventFileWriter, "add_event"),
    (EventFileWriterV2, "add_event"),
  ]
  for klass, fn_name in to_patch:
    old_fn = getattr(klass, fn_name)
    setattr(
      klass,
      fn_name,
      wraps(old_fn)(_aml_logging_wrapper(old_fn, flavor=Flavor.tensorflow)),
    )


if DEPENDENCIES_FOUND:
  if "AMLT_NO_TENSORBOARD_PATCHING" not in os.environ:
    # do not print anything here, it messes with some packages (mujoco?)
    # print("Patching tensorboard to log scalars to AML")
    register_post_import_hook(pytorch_imported, "torch.utils.tensorboard")
    register_post_import_hook(tf_imported, "tensorflow")

  # we should always (try to) run this hook, to avoid excess traffic when
  # tensorboard updates the blobfuse-located tfevents file on every write
  register_post_import_hook(tensorboard_imported, "tensorboard")
