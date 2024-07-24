import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.platform import tf_logging as logging
import numpy as np
"""
Basically EarlyStopping's restore_best_weights = True. Restore best
weights at the end of training.
"""
class RestoreBestWeights(keras.callbacks.Callback):
    
    def __init__(self, monitor="val_loss", mode="auto"):
        super().__init__()
        
        self.monitor = monitor
        self.best_weights = None
        
        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if (
                self.monitor.endswith("acc")
                or self.monitor.endswith("accuracy")
                or self.monitor.endswith("auc")
            ):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less
        
    
    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value, reference_value)
    
    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                "Restore best weights conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value
    
    def on_train_begin(self, logs=None):
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        
        
        if(self.best_weights is None):
            self.best_weights = self.model.get_weights()
        
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
        
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)