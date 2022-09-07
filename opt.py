from typing import NamedTuple
import optax

class OptConfig(NamedTuple):
    lr: float
    warmup: float
    grad_clip: float


def get_optimizer(optconf: OptConfig, beta2=0.999):
  grad_clip = optconf.grad_clip
  warmup = optconf.warmup
  lr = optconf.lr
  if optconf.optimizer == 'Adam':
    schedule = lr
    if warmup > 0:
      schedule = optax.linear_schedule(1./ warmup, lr, warmup)
    opt = optax.chain(
      optax.clip(grad_clip),
      optax.adamw(learning_rate=schedule, b1=optconf.beta1, b2=beta2, eps=optconf.eps,
                                weight_decay=optconf.weight_decay)
    )
    
  else:
    raise NotImplementedError(
      f'Optimizer {optconf.optimizer} not supported yet!')

  return opt