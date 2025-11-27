import gc
import logging
from dataclasses import dataclass
from typing import Optional

import torch

from ...aliases import PathOrStr
from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class GarbageCollectorCallback(Callback):
    """
    Disables automatic garbage collection during training and runs gen1 collection
    on a set schedule instead.

    .. important::
        This callback gets added automatically in a distributed training setting if you
        don't explicitly configure it.
        If you want to override this callback you should subclass it.
    """

    gc_interval: int = 1000
    """Interval (in steps) for running generation 1 garbage collection."""
    
    full_gc_interval: int = 5000
    """Interval (in steps) for running full garbage collection (all generations).
    This helps clean up objects that have survived to generation 2."""
    
    empty_cuda_cache: bool = True
    """Whether to empty CUDA cache during full GC to release GPU memory."""
    
    enabled: bool = True
    _start_state: Optional[bool] = None

    def pre_train(self):
        if not self.enabled:
            return
        self._start_state = gc.isenabled()
        gc.disable()
        log.info(
            f"Automatic GC disabled for training, will run gen1 GC every {self.gc_interval} steps "
            f"and full GC every {self.full_gc_interval} steps"
        )

    def post_step(self):
        if not self.enabled:
            return
        
        # Run full GC at full_gc_interval to clean up generation 2 objects
        if self.step % self.full_gc_interval == 0:
            if self.full_gc_interval > 10:
                log.info("Running full garbage collection")
            gc.collect()  # Full collection (all generations)
            if self.empty_cuda_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Run gen1 GC at gc_interval
        elif self.step % self.gc_interval == 0:
            if self.gc_interval > 10:
                log.info("Running garbage collection")
            gc.collect(1)

    def close(self):
        if not self.enabled:
            return
        if self._start_state:
            gc.enable()

    def post_checkpoint_saved(self, path: PathOrStr):
        del path
        if not self.enabled:
            return
        gc.collect()  # Full collection after checkpoint
        if self.empty_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
