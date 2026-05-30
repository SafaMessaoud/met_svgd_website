(sec2)=
# Sampling with Stein Variational Gradient Descent (SVGD)

(the-svgd-update-rule)=
## The SVGD Update Rule

<!-- Stein Variational Gradient Descent [@svgd] is a particle-based sampling algorithm that takes samples from a reference initial distribution $q^0$ and iteratively moves them in the direction of the SVGD velocity field $\phi$ until they assume positions such that they closely envelop the target. -->
Stein Variational Gradient Descent [@svgd] is a particle-based sampling algorithm that begins with samples drawn from an initial reference distribution $q^0$ and iteratively transports them via the SVGD velocity field $\phi$.
After enough iterations, the transformed particles form an empirical distribution that closely matches the target.

Formally, at iteration $l\in[0, \dots, L-1]$, the SVGD update rule is:
:::{math}
:enumerated: false
x_i^{l+1}
=
x_i^{l}
+
\epsilon
\underbrace{
    \mathbb{E}_{x_j^l \sim q^l} \left[ \kappa(x_i^l, x_j^l) \nabla_{x_j^l} \log \bar p(x_j^l) + \nabla_{x_j^l} \kappa(x_i^l, x_j^l) \right]
}_{\phi(x_i^l)},
:::
where $x_i^l$ is the $i$-th particle at iteration $l$, $\epsilon$ is the step-size, $\bar p$ is the unnormalized density, and $\kappa$ is a kernel function, most commonly the RBF kernel.

:::{figure} figures/particle_evolution.gif
:label: evolution
:class: flex flex-col items-center justify-center
Particles' evolution starting from $q^0$ and following the SVGD velocity field.
:::

:::{dropdown} Code for [](#evolution).
```python
from svgd.sampler import SVGD
from svgd.distributions import TorchDistribution
from svgd.kernels import RBF
from svgd.kernels.parameters import HeuristicKP
from svgd.lrs import ParameterLR
from svgd.callbacks import Logger

import torch
from torch.distributions import MixtureSameFamily, Categorical, Independent, Normal

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

torch.manual_seed(0)

target_distribution = TorchDistribution(
    MixtureSameFamily(
        Categorical(torch.ones(2)),
        Independent(Normal(torch.tensor([[-1.0, 0.0], [1.0, 0.0]]), 1 / 3), 1),
    )
)
initial_distribution = TorchDistribution(Independent(Normal(torch.zeros(2), 1.0), 1))
kernel = RBF(HeuristicKP("median"))
lr = ParameterLR(torch.tensor(0.5))
logger = Logger(log_x=True)
logger.activated = True
svgd = SVGD(
    target_distribution=target_distribution,
    initial_distribution=initial_distribution,
    kernel=kernel,
    lr=lr,
    callbacks=[logger],
)

n_particles = 100
n_steps = 50
x, _, _ = svgd.sample(n_particles=n_particles, n_steps=n_steps)
x = x.detach()

grid = torch.arange(-2, 2, 0.001)
xg, yg = torch.meshgrid(grid, grid, indexing="ij")
grid = torch.cat((xg.reshape(-1)[:, None], yg.reshape(-1)[:, None]), dim=-1)
zg = target_distribution.log_prob(grid).exp().view(xg.shape)

fig, ax = plt.subplots()
ax.pcolormesh(xg, yg, zg, cmap="Oranges")
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-2.0, 2.0)
ax.axis("off")
scatter = ax.scatter([], [], color="black", alpha=0.6)


def animate(frame):
    scatter.set_offsets(logger.x[frame])
    return (scatter,)


animation = FuncAnimation(
    fig,
    animate,
    len(logger.x),
    interval=1000 / 60,
    blit=False,
)
animation.save("particle_evolution.gif", fps=120)
```
:::

<!-- Importantly, the SVGD velocity field $\phi$ is carefully designed to point in the direction that minimizes the KL divergence between the updated distribution and the target distribution. In other words, each SVGD transformation moves the particles in the most "efficient" way to make them resemble the target distribution more closely. -->
A key property of SVGD is that the velocity field $\phi$ is chosen to maximally decrease the KL divergence between the particle distribution and the target.
Intuitively, each update moves the particles in the direction that most closely makes their empirical distribution resemble the target distribution.

(the-rbf-kernel)=
## The RBF Kernel

The RBF kernel is defined as
:::{math}
:enumerated: false
\kappa(x_i^l, x_j^l) = \exp\left( -\frac{1}{2\sigma^2} ||x_i^l - x_j^l||^2 \right),
:::
where $\sigma$ is the kernel bandwidth.

Using the RBF kernel, the SVGD update rule becomes:
:::{math}
:enumerated: false
x_i^{l+1}
=
x_i^{l}
+
\epsilon
\mathbb{E}
\Bigg[
\underbrace{
\kappa(x_i^l, x_j^l) \nabla_{x_j^l} \log \bar p(x_j^l)
}_{\text{drift term}}
-
\underbrace{
\frac{1}{\sigma^2} \kappa(x_i^l, x_j^l) (x_j^l - x_i^l)
}_{\text{repulsion term}}
\Bigg].
:::

In the drift term, the kernel value $\kappa(x_i^l, x_j^l)$ determines how strongly the particle $x_i^l$ is influenced by the score $\nabla_{x_j^l} \log \bar p(x_j^l)$.
This term moves particles toward regions of high probability.
In the repulsion term, the same kernel value controls how far $x_i^l$ is pushed away from $x_j^l$ along the direction $(x_j^l - x_i^l)$.
This term enforces diversity among particles, preventing them from collapsing onto a single mode.
<!-- In the drift term, the value of $\kappa(x_i^l, x_j^l)$ specifies how much $x_i^l$ should move in the direction of $\nabla_{x_j^l} \log \bar p(x_j^l)$, and in the repulsion term it modulates how far $x_i^l$ should be pushed away from $x_j^l$ along $x_i^l - x_j^l$. -->

<!-- The maximum value that $\kappa(x_i^l, x_j^l)$ can take is $1$ when $x_i^l = x_j^l$, and the minimum value it can take is $0$ as $||x_i^l - x_j^l||^2 \to \infty$.
Intuitively, $\kappa(x_i^l, x_j^l)$ measures how similar $x_i^l$ and $x_j^l$ are based on the euclidean distance, with a region of similarity controlled by $\sigma$. -->
The kernel takes its maximum value of $1$ when $x_i^l = x_j^l$, and approaches $0$ as $||x_i^l - x_j^l||^2 \to \infty$.
Intuitively, $\kappa(x_i^l, x_j^l)$ measures similarity between particles based on Euclidean distance, with the effective neighborhood size controlled by $\sigma$.

:::{figure} figures/bandwidth.svg
:label: bandwidth
:class: flex flex-col items-center justify-center
The neighborhood of $x_i^l=0$ as a function of $x_j^l$ in 1D in terms of similarity $\kappa(0, x_j^l)$ and repulsive force $\nabla_{x_j^l} \kappa(0, x_j^l)$ for different $\sigma$ values.
:::

:::{dropdown} Code for [](#bandwidth).
```python
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

torch.manual_seed(0)

bound = 10.0
x = torch.arange(-bound, bound, 0.1).unsqueeze(-1)

sigma = torch.tensor([0.5, 2.5, 5.0])
gamma = sigma.pow(2).mul(2).pow(-1)

norm = x.pow(2).sum(-1)
k = gamma.unsqueeze(-1).mul(norm).mul(-1).exp()
k_xj = gamma.mul(2).unsqueeze(-1).unsqueeze(-1).mul(k.unsqueeze(-1)).mul(x.mul(-1))

data = pd.DataFrame(
    [
        {
            "sigma": s.item(),
            "x": x[idx, 0].item(),
            "k": k[s_idx, idx].item(),
            "k_xj": k_xj[s_idx, idx, 0].item(),
        }
        for s_idx, s in enumerate(sigma)
        for idx in range(x.shape[0])
    ]
)

r, c = 1, 2
fig, ax = plt.subplots(r, c, figsize=(6.4 * c, 4.8 * r))

if c == 1:
    ax = np.array([ax])

if r == 1:
    ax = np.array([ax])

sns.lineplot(
    data,
    x="x",
    y="k",
    hue="sigma",
    ax=ax[0, 0],
    palette=sns.color_palette("tab10"),
)
ax[0, 0].grid()
ax[0, 0].set_ylabel("$\\kappa(0, x_j^l)$")
ax[0, 0].set_xlabel("$x_j^l$")

sns.lineplot(
    data,
    x="x",
    y="k_xj",
    hue="sigma",
    ax=ax[0, 1],
    palette=sns.color_palette("tab10"),
)
ax[0, 1].grid()
ax[0, 1].set_ylabel("$\\nabla_{x_j^l} \\kappa(0, x_j^l)$")
ax[0, 1].set_xlabel("$x_j^l$")

handles, labels = ax[0, 0].get_legend_handles_labels()
labels = [f"$\\sigma = {label}$" for label in labels]

ax[0, 0].legend_.remove()
ax[0, 1].legend_.remove()

legend = fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.42, 0.7),
    framealpha=1.0,
)

fig.savefig("bandwidth.svg", bbox_inches="tight")
```
:::

<!-- Setting the value of $\sigma$ is not trivial: both a very small and a very large $\sigma$ value can result in the repulsion term becoming $0$.
Classically, $\sigma$ is set according to what is called the median trick, so that $\sigma = \mathrm{median} \{ ||x_i^l - x_j^l|| \}_{i,j=0}^{M-1} / \sqrt{2 \log M}$, where $M$ is the number of particles.
This ensures that $\sum_{j \neq i} \kappa(x_i^l, x_j^l) \approx 1$.
However, it is not clear that this is something that is necessarily desirable. -->
Choosing an appropriate value for $\sigma$ is nontrivial: both extremely small and extremely large bandwidths cause the repulsion term to vanish.
A common heuristic is the median trick, which sets
$\sigma = \mathrm{median} \{ ||x_i^l - x_j^l|| \}_{i,j=0}^{M-1} / \sqrt{2 \log M}$,
where $M$ is the number of particles.
This choice roughly ensures that
$\sum_{j \neq i} \kappa(x_i^l, x_j^l) \approx 1$.

However, it is not obvious that this property is always desirable, and in practice the median trick can be somewhat brittle depending on the geometry of the target distribution.

## Derivation of the SVGD Update Rule

Suppose we want to sample from a target distribution $p$.
Given $x \sim q^0$, the idea behind SVGD is to find a velocity field $\phi$ that maximally decreases the KL divergence between the distribution $q$ of $f(x) = x + \epsilon \phi(x)$ and $p$.

Formally, we want to solve the following optimization problem:
:::{math}
:enumerated: false

\phi^*
=
\underset{\phi \in \mathcal{F}}{\mathrm{argmax}}
\,\,
- \nabla_{\epsilon} D_{KL} \left(q \,||\, p \right) |_{\epsilon=0},
:::
where $\mathcal{F}$ is a suitable family of functions.

Let $y \sim p$. One can show that $D_{KL}(q \,||\, p) = D_{KL}(q^0 \,||\, \tilde p)$, where $\tilde p$ is the distribution of $f^{-1}(y)$, assuming that $f$ is invertible.
This allows us to get a closed form expression of the gradient of $D_{KL}$ with respect to $\epsilon$, as $q^0$ is independent of $\epsilon$:
:::{math}
:enumerated: false

-\nabla_{\epsilon} D_{\mathrm{KL}}(q \,\|\, p) |_{\epsilon=0}
=
\mathbb{E}_{x\sim q^0}
\left[
\nabla_{x} \log p(x)^{\top}\,\phi(x)
+
\mathrm{Tr}(\nabla_x \phi(x))
\right].
:::

In case $\mathcal{F}$ is a subset of $\mathcal{H}^D$, where $\mathcal{H}^D$ is a Reproducing Kernel Hilbert Space (RKHS) with a corresponding kernel $\kappa(x, y)$, then, using the reproducing property of the RKHS, $\phi(x)$ can be written as $\langle \phi(\cdot), \kappa(x, \cdot) \rangle_{\mathcal{H}^D}$.
Given this, we can rewrite the gradient of the KL divergence as an inner product:
:::{math}
:enumerated: false

-\nabla_{\epsilon} D_{\mathrm{KL}}(q \,\|\, p) |_{\epsilon=0}
=
\left\langle
\phi(\cdot)
,
\mathbb{E}_{x\sim q^0}
\left[
\kappa(x, \cdot)
\nabla_{x} \log p(x)
+
\nabla_{x} \kappa(x, \cdot)
\right]
\right\rangle_{\mathcal{H}^D}.
:::

If we constrain the norm of $\phi$ in $\mathcal{H}^D$, so that $\mathcal{F} = \{ \phi : \phi \in \mathcal{H}^D \text{ s.t. } ||\phi||_{\mathcal{H}^D} \leq 1 \}$, the maximum of the inner product is achieved when $\phi$ is proportional to the second argument, hence the maximizer is:
:::{math}
:enumerated: false

\phi_{p,q^0} (\cdot)
\propto
\mathbb{E}_{x\sim q^0}
\left[
\kappa(x, \cdot)
\nabla_{x} \log p(x)
+
\nabla_{x} \kappa(x, \cdot)
\right].
:::

For a distribution known up to a normalization constant $p(x) = \bar p(x) / Z$, $\nabla_x \log p(x) = \nabla_x \log \bar p(x)$. Note, this does not depend on the normalization constant of the target density.