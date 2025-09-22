from __future__ import annotations
import dataclasses
import math
from typing import ClassVar, Dict, Type, Any, Union

import torch

Number = Union[float, torch.Tensor]

# ---------------- Registry-enabled Base ---------------- #

@dataclasses.dataclass
class BaseScheduler:
    # Class-level registry (not a dataclass field)
    __registry__: ClassVar[Dict[str, Type["BaseScheduler"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register under canonical and lowercase names
        BaseScheduler.__registry__[cls.__name__] = cls
        BaseScheduler.__registry__[cls.__name__.lower()] = cls

    # ---- common API ----
    def __call__(self, i: Number) -> Number:
        i_t = torch.as_tensor(i, dtype=torch.float32)
        if not torch.all((0.0 <= i_t) & (i_t <= 1.0)):
            raise ValueError(f"i={i} not in [0,1]")
        out = self.sub_call(i_t)
        return out.item() if isinstance(i, float) else out

    def reverse_mask_prob(self, s: Number, t: Number) -> Number:
        t_t = torch.as_tensor(t, dtype=torch.float32)
        s_t = torch.as_tensor(s, dtype=torch.float32)
        if not torch.all((0.0 <= s_t) & (s_t < 1.0) & (0.0 < t_t) & (t_t <= 1.0)):
            raise ValueError(f"(t={t}, s={s}) out of range")
        if not torch.all(s_t < t_t):
            raise ValueError(f"Require s < t elementwise, but got (t={t}, s={s})")
        out = (1 - self(s_t)) / (1 - self(t_t))
        return out.item() if isinstance(t, float) and isinstance(s, float) else out
    
    def loss_weight(self, i: Number) -> Number:
        i_t = torch.as_tensor(i, dtype=torch.float32)
        if not torch.all((0.0 < i_t) & (i_t <= 1.0)):
            raise ValueError(f"i={i} not in (0,1]")
        out = self.sub_loss_weight(i_t)
        return out.item() if isinstance(i, float) else out

    # ---- hooks implemented by subclasses ----
    def sub_call(self, i: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sub_loss_weight(self, i: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------- Implementations ---------------- #

@dataclasses.dataclass
class LinearScheduler(BaseScheduler):
    def sub_call(self, i: torch.Tensor) -> torch.Tensor:
        return 1 - i

    def sub_loss_weight(self, i: torch.Tensor) -> torch.Tensor:
        return -1.0 / i


@dataclasses.dataclass
class CosineScheduler(BaseScheduler):
    def sub_call(self, i: torch.Tensor) -> torch.Tensor:
        return 1 - torch.cos((math.pi / 2) * (1 - i))

    def sub_loss_weight(self, i: torch.Tensor) -> torch.Tensor:
        return -(math.pi / 2) * torch.tan((math.pi / 2) * (1 - i))


# ---------------- Factory helpers ---------------- #

def get_scheduler_class(name: str) -> Type[BaseScheduler]:
    """Return the scheduler class by name (case-insensitive)."""
    cls = BaseScheduler.__registry__.get(name) or BaseScheduler.__registry__.get(name.lower())
    if cls is None:
        available = sorted(k for k in BaseScheduler.__registry__ if k[0].isupper())
        raise ValueError(f"Unknown scheduler '{name}'. Available: {available}")
    return cls

def make_scheduler(name: str, **kwargs: Any) -> BaseScheduler:
    """Instantiate a scheduler by name with optional kwargs."""
    cls = get_scheduler_class(name)
    return cls(**kwargs)


# ---------------- Example usage ---------------- #

if __name__ == "__main__":
    sched = make_scheduler("LinearScheduler")
    print("Linear (float):", sched(0.5))
    print("Linear (tensor):", sched(torch.tensor([0.25, 0.5, 0.75])))

    cos = make_scheduler("CosineScheduler")
    print("Cosine (float):", cos(0.25))
    print("Cosine (tensor):", cos(torch.linspace(0.1, 0.9, 5)))
