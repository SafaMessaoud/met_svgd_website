(sec1)=
# MET-SVGD in a Nutshell

## Problem Setup

We consider a target density known only up to normalization,
:::{math}
:enumerated: false
p(x) = \frac{\bar{p}(x)}{Z},
:::
where $Z$ is intractable. Our objective is to approximate $p$ and estimate functionals of $p$, particularly its entropy.

## Problem Significance

Target distributions known only up to a normalization constant arise throughout machine learning. A prominent example is **Maximum-Entropy Reinforcement Learning (MaxEnt RL)**, which augments the standard reinforcement learning objective
:::{math}
:enumerated: false
J_{\mathrm{RL}}(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{T} r(s_t, a_t)\right]
:::
with an entropy regularization term,
:::{math}
:enumerated: false
J_{\mathrm{MaxEnt}}(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{T}\big(r(s_t, a_t) + \alpha\,\mathcal{H}(\pi(\cdot\mid s_t))\big)\right],
:::
where $\mathcal{H}(\pi(\cdot\mid s))$ denotes the policy entropy and $\alpha$ controls the reward-entropy trade-off.

Without the entropy term, the optimal solution is **a deterministic policy** that selects a single highest-value action in each state. In contrast, maximizing the entropy-regularized objective yields a **stochastic policy** that assigns probability mass to multiple high-value actions. In fact, the optimal MaxEnt policy takes the form
:::{math}
:enumerated: false
\pi(a\mid s) = \frac{\exp(Q(s,a)/\alpha)}{Z},
:::
which is an energy-based distribution with an intractable normalization constant.

The resulting stochasticity provides a significant robustness advantage. Rather than committing to a single trajectory, the agent learns a distribution over multiple high-reward behaviors. Consequently, if the environment changes at test time, the policy can exploit alternative strategies that were also assigned non-negligible probability during training.

::::{figure}
:class: flex flex-row flex-wrap items-center justify-center
:label: rl-agent

:::{figure} figures/figure_2a_maze_one_path.png
:label: rl-agent-train
Environment at train time (No obstacles).
:::

:::{figure} figures/figure_2b_maze-two-paths.png
:label: rl-agent-test
Environment at test time.
:::
A robot navigating a maze. Ref: https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/.
::::

The figure below illustrates this effect. A deterministic policy is likely to fail because it has committed to a single behavior. In contrast, a maximum-entropy policy can adapt by exploiting an alternative route that was already represented in its action distribution.

This robustness comes at a cost: evaluating the MaxEnt objective requires estimating the entropy
:::{math}
:enumerated: false
\mathcal{H}(\pi) = -\mathbb{E}_{a \sim \pi}[\log \pi(a)],
:::
yet $\pi$ is only available through an unnormalized density. Consequently, accurate and scalable entropy estimation for unnormalized distributions is a fundamental challenge in MaxEnt RL and many other machine learning applications.

## The Challenge

The core difficulty lies in the fact that the normalization constant $Z$ is unknown, which renders many standard inference methods inapplicable.
While $Z$ can be computed in closed form for certain distributions, such as Gaussians, this is generally not feasible for more complex distributions.

Some methods attempt to approximate $Z$, for example via importance sampling, but the variance of these estimates tends to grow with dimensionality, limiting their practicality.

Traditional MCMC methods (e.g., HMC, Langevin dynamics) bypass the normalization constant entirely by using the score function $\nabla_x \log p(x) = \nabla_x \log \bar{p}(x)$. However, these methods require careful hyperparameter tuning, produce only samples, and often need many iterations to yield high-quality results.

Normalizing flows, by contrast, provide both samples and densities, which allows direct estimation of $p(x)$ for a generated sample $x$, and hence also of $\mathcal{H}(p)$.
Yet, they do not directly leverage the unnormalized density $\bar{p}$, which limits their expressivity, and are prone to issues such as mode collapse.

What we ultimately seek is a method that constructs a distribution that:
- is expressive enough to capture complex, multimodal targets
- utilizes the unnormalized density $\bar{p}$
- is computationally tractable
- allows efficient sampling

## MET-SVGD

Metropolis-Hastings Stein Variational Gradient Descent (MET-SVGD) satisfies the above criteria by extending Parameterized SVGD (P-SVGD) [@s2ac], a particle-based parametric variational inference method based on SVGD [@svgd] that derives a closed-form expression of the SVGD-induced density.

MET-SVGD bridges the gap between Stein Variational Gradient Descent (SVGD) [@svgd], parametric variational inference (P-VI), and Metropolis-Hastings (MH), inheriting the strengths of each:
- the ability to approximate arbitrarily complex distributions, convergence detection, and particle efficiency form SVGD
- scalability from P-VI
- convergence guarantees from MH

:::{figure} figures/bridge.png
:label: bridge-image
:width: 300px
:class: flex flex-col items-center justify-center
MET-SVGD bridges the gap between P-VI, SVGD, and MCMC methods.
:::

```{csv-table} MET-SVGD inherits the advantages of different approximate inference methods
:label: bridge-table
:align: center
:class: flex flex-col items-center justify-center
:header: "", "P-VI", "MCMC", "SVGD", "P-SVGD", "MET-SVGD"
"Expressivity", "✗", "✓", "✓", "✓", "✓✓"
"Convergence Detection", "✓", "✗", "✓", "✓", "✓"
"Convergence Guarantees", "✗", "✓", "✗", "✗", "✓"
"Sampling Efficiency", "✓", "✗", "✓", "✓", "✓"
"Tractable Entropy", "✓", "✗", "✗", "✓", "✓"
"Parameter Efficiency", "✓", "—", "—", "✓✓", "✓✓"
```

In addition, MET-SVGD unprecedentedly scales SVGD to high-dimensional spaces, while retaining computational efficiency.

Moreover, unlike traditional approaches that rely on grid search for hyperparameter tuning, MET-SVGD enables end-to-end learning of sampler parameters via KL-divergence minimization, solving a long-standing challenge in machine learning.

Finally, MET-SVGD can be viewed as a full-rank Jacobian normalizing flow model with an adaptive number of layers controlled by a convergence check, ensuring flexibility and expressivity.