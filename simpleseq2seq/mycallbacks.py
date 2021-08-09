from keras.callbacks import Callback
import os

import numpy as np
import warnings
import logging
logger = logging.getLogger('moa.mycallbacks')
logger.setLevel('INFO')


class MyModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        export: whether to save the best model to disk. If False,
        only remember the best weights and restore the model to those
        weights when training is finished.
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath=None, monitor='val_loss', verbose=0,
                 export=True,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1,
                delete_last_file=False,
                fcn_register=None
):
        super(MyModelCheckpoint, self).__init__()
        self.export = export
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.last_filename = None
        self.delete_last_file = delete_last_file
        self.best_epoch = 0
        # callback for reporting the {key: filename} whenever
        # a model snapshot is written to disk.
        self.fcn_register = fcn_register
        
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
    
    def on_train_end(self, logs=None):
        logger.info('Model weights set to the best weights found during training (epoch {})'.format(self.best_epoch))
        self.model.set_weights(self.best_weights)
        
    def on_epoch_end_OLD(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        
                        self.best = current
                        self.best_weights = self.model.get_weights()
                        
                        if self.delete_last_file and self.export:
                            if self.last_filename:
                                try:
                                    os.remove(self.last_filename)
                                except:
                                    print("Problem deleting last checkpoint file: {}".format(self.last_filename))
                        
                        if self.export:
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)
                                self.last_filename = filepath
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' % 
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.export:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)
                        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            
            current = logs.get(self.monitor)
            
            # check if loss improved in the last epoch.
            # If yes, record the model weights
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
                last_epoch_improved = False
                return 
            else:
                if self.monitor_op(current, self.best):
                    self.best = current
                    self.best_weights = self.model.get_weights()
                    self.best_epoch = epoch
                    last_epoch_improved = True
                    
                else:
                    last_epoch_improved = False
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve from %0.5f' % 
                              (epoch, self.monitor, self.best))
            
            if self.export:
                assert self.filepath
                filepath = self.filepath.format(epoch=epoch, **logs)
                # There's one case where we don't write the new 
                # model to file. Otherwise (3 cases) we do.
                if self.save_best_only and not last_epoch_improved:
                    return
                # either we want to export whether or not there's
                # improvement, or we only want to save the best model
                # AND the latest model was an improvement. In these
                # cases, write to file.
                else:
                    if self.delete_last_file:
                        if self.last_filename:
                            try:
                                os.remove(self.last_filename)
                            except:
                                print("Problem deleting last checkpoint file: {}".format(self.last_filename))
                    
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)

                    self.last_filename = filepath
                    if self.fcn_register:
                        self.fcn_register(key="{}".format(epoch), filepath=filepath)
                    

class PeriodicModelCheckpoint(Callback):
    
    def __init__(self, filepath, period, fcn_register=None):
        self.filepath = filepath
        self.period = period
        
        # callback for reporting the {key: filename} whenever
        # a model snapshot is written to disk.
        self.fcn_register = fcn_register
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.period == 0:
            filepath = self.filepath.format(epoch=epoch, **logs)
            self.model.save(filepath, overwrite=True)
            if self.fcn_register:
                self.fcn_register(key="{}".format(epoch), filepath=filepath)
            
