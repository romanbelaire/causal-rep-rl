# Experiment Specification: Bounding Chain for Causal RL Representation

## Introduction

This document specifies the experimental protocols, architectures, environments, and evaluation metrics for engineering teams implementing the bounding chain tests in reinforcement learning. It is designed as a companion to the theoretical scope document, focusing strictly on actionable technical requirements for testing the chain of bounds from policy KL to causal representation error via gradient and convexity measures.

---

## 1. Target Environments

Experiments should be constructed in the following established RL benchmarks:

\begin{itemize}
\item \textbf{Procgen}: Diverse procedurally-generated tasks for generalization tests.
\item \textbf{MuJoCo}: Continuous control tasks with rich dynamics (e.g. Ant, Hopper, Walker2d, HalfCheetah).
\item \textbf{Minigrid}: Gridworld navigation tasks for discrete state/action spaces and interpretable policies.
\item (\textit{Optional extension}) \textbf{DMControl}: Additional continuous control environments, for wider Hessian/convexity analysis.
\end{itemize}

---

## 2. Neural Network Architectures

Experiments should systematically test multiple critic and policy architectures, to validate convexity and bound assumptions.

\begin{itemize}
\item \textbf{Critic Architectures:}
  \begin{enumerate}
  \item \textbf{ICNN} (Input Convex Neural Network): Explicitly enforce local strong convexity for value function.
  \item \textbf{Simple Feedforward NN}: Standard multilayer perceptron; test for empirical convexity near optima.
  \item \textbf{VAE-based Critic}: Use variational autoencoder representation for encoding causal features, then value function over latent space.
  \end{enumerate}
\item \textbf{Policy Architectures:}
  \begin{enumerate}
  \item \textbf{IMPALA}: Scalable RL policy implementation for large-scale multi-environment tests.
  \item \textbf{Standard MLP Policy}: Baseline for comparison.
  \end{enumerate}
\end{itemize}

**Implementation requirement**: Document all architecture hyperparameters (layer count, activation, explicit convexity constraints, latent size for VAE, etc) in config files for reproducibility.

---

## 3. Experimental Protocols

Experiments are to be conducted in the following sequence for each environment/architecture pair:

\begin{enumerate}
\item \textbf{Train baseline agents} (TRPO/PPO) with standard policy KL trust region.
\item \textbf{Train agents with representation-space trust region} (based on bounding chain; gradient/Hessian thresholding for critics).
\item \textbf{Measure all defined metrics} after convergence and during training checkpoints.
\item \textbf{Perform ablation}, switching critic architectures while holding policy fixed, and vice versa.
\end{enumerate}

---

## 4. Metrics to Collect

For each experiment, log the following quantities at regular checkpoints:

\begin{itemize}
\item \textbf{Sampled Hessian spectrum:} Compute eigenvalues of Hessian $\nabla^2 V(z)$ on critic inputs near optimal states.
\item \textbf{Fisher information index}: Estimate policy Fisher information at sampled states.
\item \textbf{KL divergence}: Aggregate policy KL between current and previous iterations.
\item \textbf{Gradient magnitude}: $||\nabla_\theta V^\pi(s)||$ per checkpoint.
\item \textbf{Value gradient difference}: $||\nabla V^{\pi'}(s) - \nabla V^{\pi}(s)||$ across policy changes.
\item \textbf{Causal prediction error}: $||Z^*(s) - Z(s)||$ with optimal representation $Z^*$ (may use known ground-truth where available, e.g. in Minigrid/Procgen).
\item \textbf{Final Policy Regret}: Empirical difference to optimal policy return, where obtainable.
\item \textbf{Occupancy measure stability}: Total variation and/or KL on state distributions.
\end{itemize}

All metrics should be stored with per-iteration and final summary statistics, then visualized (plots and tables) for direct comparison. Scripts for metric extraction must ensure compatibility across architectures and environments.

---

## 5. Reporting & Reproducibility

\begin{itemize}
\item All experimental code, configurations, and logs must be version-controlled.
\item Scripts for metric extraction and statistical analysis should be packaged for review.
\item Clear README for reproduction steps, environment setup, and architecture choices.
\item Each experiment must be accompanied by a summary table comparing all primary metrics above.
\end{itemize}

---

## 6. References

1. Richens & Everitt. "Causal World Models". ICLR 2024.
2. Schulman et al. "Trust Region Policy Optimization". ICML 2015.
3. Nabati et al. "Representation-Driven RL". ICLR 2023.
4. Your attached manuscripts: "Bounding Gradient Differences...", "Causal Bounds Formalization", "Local Convexity Theorem", "Summary Theorem & Literature Review".