(sec3)=
# P-SVGD: Particle-Based Parametric Variational Inference with SVGD

In this chapter, we show how SVGD can be transformed into a particle-based parametric variational inference algorithm called P-SVGD [@s2ac].

## Variational Inference

In variational inference, we aim to approximate an intractable target distribution $p$ with a tractable distribution $q^*$ chosen from a family of candidate distributions $\mathcal{Q}$.
Once we obtain such an approximation, we can perform inference with respect to $p$ by instead working with $q^*$, which is easier to manipulate computationally.

Formally, we solve:
:::{math}
:enumerated: false

q^*
=
\underset{q \in \mathcal{Q}}{\mathrm{argmin}}
\, D_{KL}\left( q \,||\, p \right).
:::

The choice of the family $\mathcal{Q}$ depends on the type of inference tasks that we want to perform.
If we are interested in sampling, then $\mathcal{Q}$ should be a family of distributions that are easy to sample from.
If we are interested in entropy estimation, then the elements of $\mathcal{Q}$ should have a tractable entropy, etc.

<!-- Under this view, SVGD is a particle-based variational inference algorithm, as it approximates the target with samples. -->
With this perspective, SVGD can be viewed as a particle-based variational inference method.
Instead of representing $q$ with an explicit density, SVGD represents the approximation through a set of particles.

## Parametric Variational Inference

In parametric variational inference, the family $\mathcal{Q}$ is chosen to consist of distributions with a specific functional form parameterized by a finite set of parameters.
For instance, a D-dimensional diagonal Gaussian distribution has the density:
:::{math}
:enumerated: false

p(x)
=
\frac{1}{
\sqrt{(2\pi)^{D} \det \left( \mathrm{diag}(\sigma^2) \right)}
}
\exp\left(
-\frac12
(x - \mu)^\top
\mathrm{diag}(\sigma^2)^{-1}
(x - \mu)
\right),
:::
with parameters $\mu$ and $\sigma$ representing its mean and standard deviation.

In this setting, the variational optimization problem reduces to optimizing over the parameters.
This can be done with standard numerical optimization techniques such as gradient descent.

Generally, if $\mathcal{Q}$ is a family of distributions $\{ q_\theta \}$ parameterized by $\theta$, the optimization problem becomes:
:::{math}
:label: kl-loss

\begin{align*}
\theta^*
&=
\underset{\theta}{\mathrm{argmin}}
\, D_{KL}\left( q_\theta \,||\, p \right) \\
&=
\underset{\theta}{\mathrm{argmin}}
-\mathcal{H}(q_\theta)
-\mathbb{E}_{x \sim q_{\theta}} \left[ \log p(x) \right],
\end{align*}
:::
in which case $q_\theta$ should be both easy to sample from and have a tractable entropy.

Furthermore, when the target distribution $p$ is known only up to a normalization constant, i.e. $p(x) = \bar p(x) / Z$, where $\bar p$ is the unnormalized density, it is sufficient to replace $\log p(x)$ by $\log \bar p(x)$ inside the KL objective, since $\log Z$ does not depend on $\theta$.

## Parameterized SVGD In a Nutshell

<!-- P-SVGD is a particle-based parametric variational inference algorithm based on SVGD.
It derives the closed-form expression of the SVGD-induced density of the SVGD samples after $L$ steps, $\{ q^L(x_i^L) \}_{i=0}^{M-1}$, which it then leverages to parameterize and learn the initial distribution $q^0_{\theta_1}$ by estimating $\mathcal{H}(q^L_{\theta_1})$ in Eq. [](#kl-loss) using the derived densities, accelerating convergence.
Moreover, since $q^L_{\theta_1} \approx p$, the estimate of $\mathcal{H}(q^L_{\theta_1})$ also provides an approximation of the target entropy $\mathcal{H}(p)$. -->

P-SVGD is a particle-based parametric variational inference algorithm based on SVGD.
It derives a closed-form expression of the SVGD-induced density of the SVGD samples after $L$ steps, $\{ q^L(x_i^L) \}_{i=0}^{M-1}$, which it uses to:
- estimate the entropy of the target $\mathcal{H}(p)$
- learn the initial distribution, which it parametrizes as $q^0_{\theta_1} = \mathcal{N}(\mu_{\theta_1}, \mathrm{diag}(\sigma^2_{\theta_1}))$, using Eq. [](#kl-loss), to accelerate convergence

Moreover, as a divergence control heuristic, P-SVGD truncates particles that do not satisfy $|x_i^l - \mu_{\theta_1}| \leq 3 \sigma_{\theta_1}$.
Intuitively, if the initial distribution covers the target modes, particles that stray too from it are likely to have diverged.

## Derivation of the SVGD-induced Density

:::{figure} figures/density_evolution.gif
:label: density-evolution
:class: flex flex-col items-center justify-center
Density' evolution starting from $q^0$ and following the SVGD velocity field.
:::

:::{dropdown} Code for [](#density-evolution).
```python
from svgd.sampler import SVGD
from svgd.distributions import TorchDistribution
from svgd.kernels import RBF
from svgd.kernels.parameters import HeuristicKP
from svgd.lrs import ParameterLR
from svgd.callbacks import Logger

import torch
from torch.distributions import MixtureSameFamily, Categorical, Independent, Normal

import matplotlib
import pandas as pd
import seaborn as sns
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

bound = 2.5

grid = torch.arange(-bound, bound, 0.01)
xg, yg = torch.meshgrid(grid, grid, indexing="ij")
grid = torch.cat((xg.reshape(-1)[:, None], yg.reshape(-1)[:, None]), dim=-1)
zg = target_distribution.log_prob(grid).exp().view(xg.shape)

fig, ax = plt.subplots()
ax.pcolormesh(xg, yg, zg, cmap="Oranges")
ax.set_xlim(-bound, bound)
ax.set_ylim(-bound, bound)
ax.axis("off")
scatter = ax.scatter([], [], color="black", alpha=0.6)


def animate(frame):
    for artist in ax.collections:
        if isinstance(artist, matplotlib.contour.QuadContourSet):
            artist.remove()

    scatter.set_offsets(logger.x[frame])
    sns.kdeplot(
        x=logger.x[frame][:, 0],
        y=logger.x[frame][:, 1],
        ax=ax,
        alpha=0.4,
        color="black",
    )

    return (scatter,)


animation = FuncAnimation(
    fig,
    animate,
    len(logger.x),
    interval=1000 / 60,
    blit=False,
)
animation.save("density_evolution.gif", fps=120)
```
:::

Suppose that, after $l$ SVGD steps, the particle $x^l$ is distributed according to $q^l$.
Then, using the change of variable formula for densities (CVF), the distribution of $x^{l+1} = x^l + \epsilon \phi(x^l)$, where $\phi$ is the [SVGD velocity field](#the-svgd-update-rule), is:

:::{aside} Change of Variable Formula (CVF)
:label: cvf-aside
Given a random variable $X \in \mathbb{R}^D$ and an invertible, deterministic transformation $T:\mathbb{R}^D\to\mathbb{R}^D$, the CVF gives the expression of the density of $Y=T(X)$: $q_Y(T(x)) = q_X(x)\left|\det \nabla_{x}T(x)\right|^{-1}$.
:::

:::{math}
:label: cvf

\begin{align*}
q^{l+1}(x^{l+1})
&=
q^{l}(x^{l})
\left|\det \nabla_{x^l} \left( x^l + \epsilon \phi(x^l) \right)\right|^{-1}
\\
&=
q^{l}(x^{l})
\left|\det \left( I + \epsilon \nabla_{x^l} \phi(x^l) \right)\right|^{-1}.
\end{align*}
:::

However, to apply the CVF, the SVGD transformation must be invertible.
To ensure this, @s2ac show that it is sufficient to choose $\epsilon \ll \sigma$, where $\sigma$ is the [RBF kernel bandwidth](#the-rbf-kernel).

In practice, it is easier to work with the log of the density, also called the log-likelihood:
:::{math}
:enumerated: false
\log q^{l+1}(x^{l+1})
=
\log q^{l}(x^{l})
-
\log \left|\det \left( I + \epsilon \nabla_{x^l} \phi(x^l) \right)\right|.
:::

Unfortunately, this expression involves the log of the determinant of a jacobian, which is expensive to compute.
To avoid this, @s2ac leverage the corollary of Jacobi's formula and obtain:

:::{aside} Corollary of Jacobi's Formula (CJF)
For an invertible square matrix $A$, the CJF allows us to rewrite its log-determinant as
$\log \left( \det A \right) = \mathrm{Tr} \left( \log A \right)$,
which, if
$||A-I||_{\infty} \ll 1$,
becomes
$\mathrm{Tr} \left( \log A \right) = \mathrm{Tr}\left( A - I \right) + \mathcal{O}(\epsilon^2)$.
:::

:::{math}
:enumerated: false
\log q^{l+1}(x^{l+1})
=
\log q^{l}(x^{l})
-
\epsilon \mathrm{Tr}\left( \nabla_{x^l} \phi(x^l) \right)
+
\mathcal{O}(\epsilon^2),
:::
if $\epsilon ||\nabla_{x^l} \phi(x^l)||_{\infty} \ll 1$, which, in practice, they assume is satisfied by setting $\epsilon$ to a small value.

For samples $\{ x_i^l \}_{i=0}^{M-1} \sim q^l$, the trace term evaluated at $x_i^l$ is:
:::{math}
:enumerated: false

\begin{align*}
\mathrm{Tr}\left( \nabla_{x_i^l} \phi(x_i^l) \right)
&=
\frac{1}{M}
\sum_{j=0}^{M-1}
\left[
\nabla_{x_i^l} \kappa(x_i^l, x_j^l)^\top
\nabla_{x_j^l} \log \bar p(x_j^l)
+
\mathrm{Tr}\left( \nabla_{x_i^l} \nabla_{x_j^l} \kappa(x_i^l, x_j^l) \right)
\right]
\\
&+
\frac{1}{M}
\mathrm{Tr}\left( \nabla_{x_i^l}^2 \log \bar p(x_i^l) \right).
\end{align*}
:::
When $\kappa$ is the RBF kernel, the first term in the above expression can be efficiently computed using only vector dot products.
However, the second term is computationally expensive because it involves computing the trace of a hessian, which has $\mathcal{O}(D^2)$ complexity.

To address this, @s2ac proposed estimating the expectation in $\phi$ using $\{ x_j^l \}_{j=0}^{M-1} - \{ x_i^l \}$, which removes the trace-of-hessian term and yields:
:::{math}
:enumerated: false

\mathrm{Tr}\left( \nabla_{x_i^l} \phi(x_i^l) \right)
=
\frac{1}{M}
\sum_{\substack{j=0 \\ j \neq i}}^{M-1}
\left[
\nabla_{x_i^l} \kappa(x_i^l, x_j^l)^\top
\nabla_{x_j^l} \log \bar p(x_j^l)
+
\mathrm{Tr}\left( \nabla_{x_i^l} \nabla_{x_j^l} \kappa(x_i^l, x_j^l) \right)
\right]
.
:::