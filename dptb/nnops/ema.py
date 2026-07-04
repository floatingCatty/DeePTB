import torch
import logging
from contextlib import contextmanager

log = logging.getLogger(__name__)


class ExponentialMovingAverage:
    """Maintains an exponential moving average of model parameters.

    shadow <- decay * shadow + (1 - decay) * param  after every optimizer step.

    The averaged weights are used for validation and checkpointing (see Trainer /
    Saver), while raw weights keep training. This removes the single-iteration
    noise from best-checkpoint selection and typically lowers eval loss late in
    training (standard practice in NequIP / MACE / Allegro).
    """

    def __init__(self, parameters, decay: float, use_num_updates: bool = True):
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"ema decay must be in [0, 1], got {decay}")
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        parameters = [p for p in parameters if p.requires_grad]
        self._params = parameters
        self.shadow = [p.detach().clone() for p in parameters]
        self._backup = None

    def update(self):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            # warmup: effective decay ramps up so early steps are not dominated
            # by the random initialization
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            for s, p in zip(self.shadow, self._params):
                s.add_(one_minus_decay * (p - s))

    def copy_to(self, parameters=None):
        parameters = self._params if parameters is None else [p for p in parameters if p.requires_grad]
        with torch.no_grad():
            for s, p in zip(self.shadow, parameters):
                p.copy_(s)

    @contextmanager
    def average_parameters(self):
        """Context manager: model temporarily holds the EMA weights."""
        self._backup = [p.detach().clone() for p in self._params]
        self.copy_to()
        try:
            yield
        finally:
            with torch.no_grad():
                for b, p in zip(self._backup, self._params):
                    p.copy_(b)
            self._backup = None

    def state_dict(self):
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        shadow = state_dict["shadow"]
        if len(shadow) != len(self.shadow):
            raise ValueError(
                f"EMA state has {len(shadow)} tensors but model has {len(self.shadow)} "
                "trainable parameters; cannot restore."
            )
        for i, (s, p) in enumerate(zip(shadow, self._params)):
            if tuple(s.shape) != tuple(p.shape):
                raise ValueError(
                    f"EMA shadow tensor {i} has shape {tuple(s.shape)} but the model parameter "
                    f"has shape {tuple(p.shape)}; cannot restore."
                )
        self.shadow = [s.detach().clone().to(p.device, p.dtype) for s, p in zip(shadow, self._params)]
