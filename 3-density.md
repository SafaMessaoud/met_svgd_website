(sec3)=
# The SVGD-Induced Density

As a particle-based variational inference method, SVGD [@svgd] evolves a set of $M$ particles $\{ x_i^0 \}_{i=0}^{M-1}$ sampled from an initial distribution $q^0$ to collectively match any arbitrarily complex target distribution $p$, as long as it is possible compute the score of $p$.
However, SVGD does not provide an explicit expression for the density it induces: while we can generate samples, we do not know the value of the corresponding probability density at those sampled points.
That is, if we evolve $\{ x_i^0 \}_{i=0}^{M-1}$ for $L$ steps according to the SVGD update rule, we would obtain $\{ x_i^L \}_{i=0}^{M-1} \sim q^L$ with $q^L \approx p$, but we wouldn't know the value of $q^L(x_i^L)$ for a given $x_i^L$.
This is problematic when we are interested in downstream tasks such as likelihood-based evaluation, uncertainty quantification, or entropy estimation, as these require access to the density itself rather than just samples.

(sec3-1)=
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
To ensure this, MET-SVGD adapts a sufficient conditions for the invertibility of transformations of the form $f(x)=x+g(x)$ from @behrinv to SVGD.
Namely, $f(x^l) = x^l + \epsilon \phi(x^l)$ is invertible if $\epsilon \sup_{x^l} ||\nabla_{x^l} \phi(x^l)||_2 < 1$, where $||\nabla_{x^l} \phi(x^l)||_2$ denotes the spectral norm of $\nabla_{x^l} \phi(x^l)$.

:::{aside} Spectral Norm
The spectral norm of a real-valued matrix $A$ is defined as $||A||_2 = \sigma_\text{max}(A)$, where $\sigma_\text{max}(A)$ is the largest singular value of $A$.
An equivalent and more intuitive definition is $||A||_2 = \max_{||v||=1} ||Av||$, i.e. $||A||_2$ measures the maximum amount by which $A$ can stretch a unit vector.
:::

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
To avoid this, under the condition
$\epsilon |\lambda_\text{max}(\nabla_{x^l} \phi(x^l))| < 1$,
MET-SVGD accurately estimates
$\log \left|\det \left( I + \epsilon \nabla_{x^l} \phi(x^l) \right)\right|$
by
$\epsilon \mathrm{Tr}\left( \nabla_{x^l} \phi(x^l) \right)$,
giving:

:::{math}
:enumerated: false
\log q^{l+1}(x^{l+1})
\approx
\log q^{l}(x^{l})
-
\epsilon \mathrm{Tr}\left( \nabla_{x^l} \phi(x^l) \right).
:::

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

(sec3-tr)=
One way to altogether bypass computing this term is to estimate the expectation in $\phi(x_i^l)$ using $\{ x_i^l \}_{i=0}^{M-1} - \{ x_i^ l\}$.
However, this turns out to be suboptimal in the finite particle case.
Instead, MET-SVGD efficiently estimates it as:

:::{aside}
- $(i)$ Using Hutchinson's estimator [@hutch]
- $(ii)$ Using the double differentiation trick [@ddt]
:::

:::{math}
:enumerated: false

\begin{align*}
\frac1M\mathrm{Tr}\left( \nabla_{x_i^l}^2 \log \bar p(x_i^l) \right)
&\stackrel{(i)}{=}
\frac1M\mathbb{E}_{v \sim p_v}\left[
v^\top
\nabla_{x_i^l}^2 \log \bar p(x_i^l)
v
\right] \\
&\stackrel{(ii)}{=}
\frac1M\mathbb{E}_{v \sim p_v}\left[
\nabla_{x_i^l} \left( v^\top \nabla_{x_i^l} \log \bar p(x_i^l) \right)
v
\right] \\
&\approx
\frac1{MV}
\sum_{k=0}^{V-1}
\nabla_{x_i^l} \left(
v_k^\top \nabla_{x_i^l} \log \bar p(x_i^l)
\right)
v_k,
\end{align*}
:::

where $v \sim p_v$ satisfy $\mathbb{E}_{v \sim p_v}[v]=0$ and $\mathbb{E}_{v \sim p_v}[vv^\top]=I$, and $V$ is the number of $v_k$ samples.

Since the estimator is weighted by $\frac1M$, its variance is greatly reduced, and, in practice, only one $v$ is sufficient.

(sec3-eps)=
## Unifying the Step-Size Conditions

The correctness of the previous derivation depends on two conditions:
- $\epsilon \sup_{x^l} ||\nabla_{x^l} \phi(x^l)||_2 < 1$ to ensure that the SVGD transformation is invertible
- $\epsilon |\lambda_\text{max}(\nabla_{x^l} \phi(x^l))| < 1$ for the approximation of $\log \left|\det \left( I + \epsilon \nabla_{x^l} \phi(x^l) \right)\right|$ to be accurate

While these are separate conditions, they can be unified by considering the order relation between the spectral norm of a real-valued square matrix $A$ and the magnitude of $\lambda_\text{max}(A)$.

According to @matrixbounds, for $A \in \mathbb{R}^{d \times d}$:
:::{math}
:enumerated: false

|\lambda_i(A)|
\leq
\sigma_i(A)
\leq
\sqrt{ \mathrm{Tr}(A^\top A) }
\quad
\forall i \in [1 \dots d],
:::
where $\lambda_i(A)$ and $\sigma_i(A)$ are the $i$-th eigenvalue and singular value of $A$, respectively.

Given that $||\nabla_{x^l} \phi(x^l)||_2 = \sigma_\text{max}(\nabla_{x^l} \phi(x^l))$, we have:
:::{math}
:enumerated: false

\epsilon |\lambda_\text{max}(\nabla_{x^l} \phi(x^l))|
\leq
\epsilon \sup_{x^l} ||\nabla_{x^l} \phi(x^l)||_2
\leq
\epsilon \sup_{x^l} \sqrt{ \mathrm{Tr}\left( \nabla_{x^l} \phi(x^l)^\top \nabla_{x^l} \phi(x^l) \right) }
.
:::

Therefore, in order to satisfy both conditions, it is sufficient to choose the step-size such that:
:::{math}
:enumerated: false

\epsilon
<
\left(
\sup_{x^l}
\sqrt{
\mathrm{Tr}\Big( \nabla_{x^l} \phi(x^l)^\top \nabla_{x^l} \phi(x^l) \Big)
}
\right)^{-1}
.
:::

Note that
$\mathrm{Tr}\Big( \nabla_{x^l} \phi(x^l)^\top \nabla_{x^l} \phi(x^l) \Big)$
can be efficiently computed using only vector dot products and first-order derivatives, similarly to
$\mathrm{Tr}\left( \nabla_{x^l} \phi(x^l) \right)$.
And, in practice, MET-SVGD solves $\sup_{x^l}$ by taking the maximum over particles $\{ x_i^l \}_{i=0}^{M-1}$ at iteration $l$.