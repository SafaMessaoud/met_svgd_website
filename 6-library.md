(sec6)=
# Library

The library is available on GitHub: [SafaMessaoud/MET-SVGD-Variational-Inference-With-SVGD](https://github.com/SafaMessaoud/MET-SVGD-Variational-Inference-With-SVGD).

## Quickstart

```python
from svgd import SVGD
from svgd.distributions import TorchDistribution, Gaussian
from svgd.kernels import RBF
from svgd.kernels.parameters import ParameterKP
from svgd.lrs import ParameterLR

import torch
from torch.optim.adam import Adam
from torch.distributions import MultivariateNormal

from tqdm import tqdm
import matplotlib.pyplot as plt

from math import log, sqrt

device = "cuda:3"
d = 10

# initialize the target distribution
target_distribution = TorchDistribution(
    MultivariateNormal(torch.zeros(d).to(device), torch.eye(d).to(device))
)

# initialize the initial distribution
initial_distribution = Gaussian(torch.ones(d).mul(2), torch.ones(d))

# initialize the kernel
kernel = RBF(
    ParameterKP(
        torch.tensor(1.0).log(),
        lambda x: x.clamp(log(1e-2), log(sqrt(d / 2))).exp(),
    )
)

# initialize the learning rate
lr = ParameterLR(torch.tensor(0.1).log(), lambda x: x.exp())

# initialize the SVGD object
svgd = SVGD(
    target_distribution=target_distribution,
    initial_distribution=initial_distribution.requires_grad_(True),
    kernel=kernel.requires_grad_(True),
    lr=lr.requires_grad_(True),
    divergence_control="metropolis-hastings",
    bound_lr=True,
    ij_term_density=True,
).to(device)

# initialize the optimizer
optimizer = Adam(svgd.parameters(), 5e-2)

# for statistic collection
loss_hist = []
entropy_hist = []

# main training loop
for _ in tqdm(range(200)):
    # get samples and their log densities
    x, mask, log_q = svgd.sample_with_log_q(n_particles=100, n_steps=100)
    
    # compute the entropy
    entropy = log_q.mul(mask).sum(-1).div(mask.sum(-1)).mul(-1)

    # compute the kld loss
    log_p = target_distribution.log_prob(x).mul(mask).sum(-1).div(mask.sum(-1))
    loss = entropy.add(log_p).mul(-1)

    # update parameters
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss_hist.append(loss.item())
    entropy_hist.append(entropy.item())
```

## The SVGD Class

The signature of the `__init__` method of the SVGD class is:
```python
def __init__(
    self: Self@SVGD,
    target_distribution: TargetDistribution,
    initial_distribution: InitialDistribution,
    kernel: Kernel,
    lr: LR,
    divergence_control: DivergenceControl = None,
    bound_lr: bool = False,
    ij_term_density: bool = False,
    track_convergence: bool = False,
    callbacks: List[Callback] = [],
    leaky_lr_clamp: bool = False
) -> None: ...
```

`target_distribution` and `initial_distribution` are explained in depth in the next section.

Using the RBF kernel is explained in the Quickstart example, and creating custom bandwidths and step-sizes is shown with examples in [Section 5](#sec5-ck).

`divergence_control` can be either of `None`, or `metropolis-hastings`.

`bound_lr` controls whether the step-size condition is applied.

`ij_term_density` controls whether the [trace-of-Hessian term](#sec3-tr) is computed.

`track_convergence` indicates whether or not to compute `self.stein_identity`. Useful with `Callback`s.

`callbacks` is an array of `Callback`s. This is explained in [Section 6](#sec6-cb).

`leaky_lr_clamp` indicates whether the `lr_bound` is a hard `clamp` or not. Hard clamping kills gradient flow at the extremes.

## Custom Distributions

```python
from svgd.distributions import TargetDistribution, InitialDistribution

import torch
from torch import Tensor
from torch.nn import Module, Parameter

import matplotlib.pyplot as plt

from math import log, pi
```

To create a custom target distribution, all that is required is to create a class that implements the `TargetDistribution` interface, which can be found in `svgd.distributions`. Similarly for initial distributions.

The following are example target and initial diagonal GMM distributions. I will assume that `means` and `stdevs` both have the shape `(n_components, n_dimensions)` and `weights` is `(n_components,)`. But, of course, as long as your implementation adheres to the interface, your distributions can be as complicated as you require.

```python
class TargetDiagonalGMM(TargetDistribution):
    def __init__(self, means: Tensor, stdevs: Tensor, weights: Tensor):
        assert len(means.shape) == 2
        assert len(weights.shape) == 1
        assert means.shape == stdevs.shape
        assert means.shape[-2] == weights.shape[-1]
        assert stdevs.gt(0.0).all()

        self.means = means
        self.stdevs = stdevs
        self.weights = weights.softmax(-1)

    def log_prob(self, x: Tensor):
        # the input tensor `x` has shape (b1, ..., bm, n, d)
        # the bi are optional, hence len(x.shape) is at least 2

        vars = self.stdevs.pow(2)

        return (
            x.unsqueeze(-2)
            .sub(self.means)
            .pow(2)
            .div(vars)
            .add(vars.log())
            .add(log(2 * pi))
            .sum(-1)
            .div(-2)
            .add(self.weights.log())
            .logsumexp(-1)
        )
```

```python
class InitialDiagonalGMM(InitialDistribution, Module):
    def __init__(self, means: Tensor, stdevs: Tensor, weights: Tensor):
        Module.__init__(self)

        assert len(means.shape) == 2
        assert len(weights.shape) == 1
        assert means.shape == stdevs.shape
        assert means.shape[-2] == weights.shape[-1]
        assert stdevs.gt(0.0).all()

        self.means = Parameter(means)
        self.log_stdevs = Parameter(stdevs.log())
        self.logits_weights = Parameter(weights.log_softmax(-1))

    @property
    def stdevs(self):
        return self.log_stdevs.clamp(-5, 2).exp()

    def rsample(self, n_particles: int) -> Tensor:
        stdevs = self.stdevs
        weights = self.logits_weights.softmax(-1)

        idx = torch.multinomial(weights, n_particles, True)

        return (
            torch.randn(n_particles, self.means.shape[-1], device=self.means.device)
            .mul(stdevs[idx])
            .add(self.means[idx])
        )

    def log_prob(self, x: Tensor):
        vars = self.stdevs.pow(2)
        log_weights = self.logits_weights.log_softmax(-1)

        return (
            x.unsqueeze(-2)
            .sub(self.means)
            .pow(2)
            .div(vars)
            .add(vars.log())
            .add(log(2 * pi))
            .sum(-1)
            .div(-2)
            .add(log_weights)
            .logsumexp(-1)
        )
```

(sec5-ck)=
## Custom Kernel Bandwidths and Step-Sizes

```python
from svgd.lrs import LR
from svgd.kernels.parameters import KP
from svgd.states import StateForKernel, StateForLR

from torch import Tensor
from torch.nn import Module, Parameter, Linear

from typing_extensions import Union, List
```

The RBF kernel is defined as $$k(x,y) = \exp\left(-\frac{1}{2\sigma^2}||x-y||^2\right).$$

In this library, $\sigma$ is defined as a class that implements the `KP` interface available at `svgd.kernels.parameters`. The interface only requires that a forward method be implemented. It takes `state: StateForKernel` and `**kwargs` as attributes. `state` lists the available quantities on the `SVGD` object.

`sigma` must be a `(b1, ..., bm)` tensor.

Suppose we decide that we are always going to perform `s` SVGD steps, then a straightforward sigma implementation would be a `(s,)` tensor learnable via gradient descent.

```python
class CustomSigma(KP, Module):
    def __init__(self, sigma: Tensor):
        assert sigma.dim() == 1
        assert sigma.ge(0.0).all()

        Module.__init__(self)

        self.log_sigma = Parameter(sigma.add(1e-5).log())

    def forward(self, state: StateForKernel, **kwargs):
        return self.log_sigma[state.step].clamp(-10, 10).exp()
```

In other cases, your $\sigma$ will be parametrized by a neural network (e.g. reinforcement learning).
In this case, we can't just inspect the weights to get the current sigma values as we would do with a `Parameter` $\sigma$, since every evaluation gives a different one, but we can track the values the network outputted throughout the SVGD steps.
To that end, you could implement a logger wrapper class.

```python
class SigmaNN(KP, Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        Module.__init__(self)

        self.obs: Union[None, Tensor] = None  # (b1, ..., bm, obs_dim)

        self.l1 = Linear(obs_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, hidden_dim)
        self.log_sigma = Linear(hidden_dim, 1)

    def set_observation(self, obs: Tensor):
        self.obs = obs

    def forward(self, state: StateForKernel, **kwargs):
        if self.obs is None:
            raise Exception()

        l1 = self.l1.forward(self.obs).relu()
        l2 = self.l2.forward(l1).relu()
        sigma = self.log_sigma.forward(l2).clamp(-10, 10).exp()
        
        self.obs = None

        return sigma.squeeze(-1)  # must be (b1, ..., bm)
```

```python
class SigmaWithHist(KP, Module):
    def __init__(self, sigma: KP):
        Module.__init__(self)

        self.sigma = sigma
        self.log_sigma = False
        self.sigma_hist: List[Tensor] = []

    def forward(self, state: StateForKernel, **kwargs):
        sigma = self.sigma.forward(state, **kwargs)

        if not self.log_sigma:
            return sigma
        
        if state.step == 0 and len(self.sigma_hist) > 0:
            print("Did you forget to reset the sigma history?")
            # raise Exception()
        
        self.sigma_hist.append(sigma)
```

This can then be used as follows:
```python
sigma = SigmaWithHist(SigmaNN(100, 64))
# ...
svgd = SVGD(kernel=RBF(sigma), ...)

for i in range(n_iter):
    sigma.log_sigma = i % 100 == 0
    obs, ... = ...
    sigma.set_observation(obs)
    x, mask, log_q = svgd.sample_with_log_q(n_particles, n_steps)
    # ... etc
```

Dealing with the learning rate is exactly the same. The interface that should be implemented is `LR`, which can be found at `gamma.svgd.lrs`.
The learning, too, must be a `(b1, ..., bm)` tensor.
```python
class CustomLR(LR, Module):
    def __init__(self, lr: Tensor):
        assert lr.dim() == 1
        assert lr.ge(0.0).all()

        Module.__init__(self)

        self.log_lr = Parameter(lr.add(1e-5).log())

    def forward(self, state: StateForLR):
        return self.log_lr[state.step].clamp(-10, state.x.shape[-2]).exp()
```

(sec6-cb)=
## Callbacks

`Callback` is an abstract `class` defined as:
```python
class Callback:
    def on_initialization_done(self, state: StateOnInitializationDone):
        pass

    def on_sampling_iteration_started(self, state: StateOnSamplingIterationStarted):
        pass

    def on_score_computed(self, state: StateOnScoreComputed):
        pass

    def on_proposal_computed(self, state: StateOnProposalComputed):
        pass

    def on_sampling_iteration_done(self, state: StateOnSamplingIterationDone):
        pass

    def on_sampling_done(self, state: StateOnSamplingDone):
        pass
```

`Callback`s are executed at specified stages identified by states during a given iteration.
For example, the state `StateOnProposalComputed` means that the loop is at the stage where the proposal has been computed and is available in the `state` variable.

Importantly, we expose `f_attach_iteration`, which controls whether or not to detach the gradients during a given iteration (it is set to `False` by default), and `f_break_sampling_loop`, which controls whether to break the sampling loop (e.g. based on the stein identity).

## States

All states are listed under `svgd.states`. Note that they build upon one another.

```python
class Flags(Protocol):
    ...
    f_attach_iteration: bool
    f_break_sampling_loop: bool
```

```python
class StateOnInitializationDone(Flags, Protocol):
    n_steps: int
    cur_n_particles: Tensor

    x: Tensor
    mask: Tensor
    log_q: Tensor
    mask_off_diagonal: Tensor
```

```python
class StateOnSamplingIterationStarted(StateOnInitializationDone, Protocol):
    step: int
```

```python
class StateOnScoreComputed(StateOnSamplingIterationStarted, Protocol):
    log_p: Tensor
    score: Tensor
```

```python
class StateOnScoreValuesComputed(StateOnScoreComputed, Protocol):
    v: Tensor
    grad_v_score: Tensor
    tr_j_score: Tensor
```

```python
class StateForKernel(StateOnScoreValuesComputed, Protocol):
    pass
```

```python
class StateOnKernelValuesComputed(StateForKernel, Protocol):
    k: Tensor
    k_xj: Tensor
    k_xi: Tensor
    tr_k_xj_xi: Tensor
    score_t_k_xk_xi_k_xi: Tensor
    tr_k_xj_xi_t_k_xk_xi: Tensor
    v_t_k_xj_xi_grad_v_score: Tensor
```

```python
class StateOnPhiValuesComputed(StateOnKernelValuesComputed, Protocol):
    s_1: Tensor
    s_2: Tensor
    phi: Tensor

    tr_j_phi: Tensor

    tr_a_t_a: Tensor
    tr_a_b_t: Tensor
    tr_b_b_t: Tensor
    tr_j_phi_t_j_phi: Tensor
```

```python
class StateForLR(StateOnPhiValuesComputed, Protocol):
    pass
```

```python
class StateOnLRComputed(StateForLR, Protocol):
    lr_value: Tensor
    lr_bound: Tensor
    corrected_lr_value: Tensor
```

```python
class StateOnProposalComputed(StateOnLRComputed, Protocol):
    log_abs_det_j: Tensor
    x_svgd: Tensor
```

```python
class StateOnDivergenceControlled(StateOnProposalComputed, Protocol):
    log_p_acc: Tensor
    p_acc: Tensor
    p_rej: Tensor
    mask_mh: Tensor
```

```python
class StateOnSamplingIterationDone(StateOnDivergenceControlled, Protocol):
    stein_identity: Tensor
```

```python
class StateOnSamplingDone(StateOnSamplingIterationDone, Protocol):
    pass
```