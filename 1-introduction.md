(sec1)=
# MET-SVGD in a Nutshell

## Motivation and Problem Setup

We consider the setting where we only have access to a target density $p$ known up to a normalization constant
:::{math}
:enumerated: false
p(x) = \frac{\bar{p}(x)}{Z},
:::
where $\bar{p}$ is the unnormalized density and $Z$ is the normalization constant, which is generally intractable to compute.

Our aim is to perform a range of inference tasks with respect to $p$, such as:
- generating high-quality samples
- estimating $p(x)$ for a given sample $x$
- estimating the entropy of $p$

To accomplish these goals, in practice, we typically rely on a parameterized sampling mechanism, such as a learned sampler, a neural transformation, or a parameterized particle update rule.
The quality of inference hinges on how well this mechanism can represent the target density.

## Problem Significance

This setup arises frequently in machine learning.
A motivating example is maximum-entropy reinforcement learning (MaxEnt RL), where policies are defined through unnormalized energy-based distributions over actions.

Policies trained under the maximum-entropy reinforcement learning framework tend to be more robust, as the agent learns to capture multiple modes of high-reward behavior rather than committing to a single deterministic trajectory.
Consequently, if the environment or the state is perturbed at test time, the agent is more likely to recover by exploiting alternative high-reward strategies.

::::{figure}
:class: flex flex-row flex-wrap items-center justify-center
:label: rl-agent

:::{figure} figures/figure_2a_maze_one_path.png
:label: rl-agent-train
Environment at train time.
:::

:::{figure} figures/figure_2b_maze-two-paths.png
:label: rl-agent-test
Environment at test time.
:::

https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/.
::::

This is illustrated in the figure above, where the test time environment includes an additional obstacle that the agent hasn't seen during training.
A standard RL agent that has learned a deterministic policy would not be able to reach the goal, whereas a MaxEnt RL agent would be able to find the lower passage towards the goal.

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