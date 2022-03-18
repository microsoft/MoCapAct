from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
from gym import spaces
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.distributions.kl import kl_divergence

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.vec_env import VecNormalize


class PPOBC(PPO):
    """
    PPO algorithm with an additional behavior cloning (BC) loss.
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        kl_coef: float = 0.,
        vf_coef: float = 0.5,
        bc_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        bc_dataloader: Optional[torch.utils.data.DataLoader] = None,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=0.,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model
        )

        self.kl_coef = kl_coef
        self.bc_coef = bc_coef
        self.bc_dataloader = bc_dataloader
        if self.bc_dataloader:
            self.bc_generator = iter(self.bc_dataloader)

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["bc_dataloader", "bc_generator"]

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        pg_losses, value_losses = [], []
        bc_losses = []
        embedding_kls = []
        bc_kls, bc_mses, bc_embed_stds, bc_embed_dists = [], [], [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                values, log_prob, embed_gaussian = self.policy.evaluate_actions(rollout_data.observations, actions)
                prev_embed = rollout_data.observations['embedding']
                if isinstance(self.env, VecNormalize) and 'embedding' in self.env.norm_obs_keys:
                    prev_embed = torch.tensor(self.env._unnormalize_obs(prev_embed.cpu().numpy(), self.env.obs_rms['embedding']), device=prev_embed.device)
                prior_gaussian = Independent(Normal(self.policy.embedding_correlation*prev_embed, self.policy.embedding_std_dev), -1)
                kl_div = kl_divergence(embed_gaussian, prior_gaussian).mean()
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                embedding_kls.append(kl_div.item())

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Behavior cloning loss
                if self.bc_dataloader:
                    try:
                        # sample the batch
                        obs_bc, act_bc, weights = next(self.bc_generator)
                    except StopIteration:
                        # restart the generator if the previous generator is exhausted
                        self.bc_generator = iter(self.bc_dataloader)
                        obs_bc, act_bc, weights = next(self.bc_generator)
                    #obs_bc = self.env.normalize_obs({k: v.numpy() for k, v in obs_bc.items()})
                    obs_bc = {k: torch.tensor(v, device=self.device) for k, v in obs_bc.items()}
                    act_bc = act_bc.to(self.device)
                    weights = weights.to(self.device)
                    bc_loss, kl_bc, mse_bc, embed_std_bc, embed_dist_bc = self.policy.compute_bc_loss(obs_bc, act_bc, weights)
                    bc_losses.append(bc_loss.item())
                    bc_kls.append(kl_bc.item())
                    bc_mses.append(mse_bc.item())
                    bc_embed_stds.append(embed_std_bc.item())
                    bc_embed_dists.append(embed_dist_bc.item())
                else:
                    bc_loss = 0.
                    bc_losses.append(bc_loss)

                loss = policy_loss + self.kl_coef * torch.mean(kl_div) + self.vf_coef * value_loss + self.bc_coef * bc_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/embedding_kl", np.mean(embedding_kls))
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        self.logger.record("behavior_cloning/loss", np.mean(bc_losses))
        self.logger.record("behavior_cloning/mse", np.mean(bc_mses))
        self.logger.record("behavior_cloning/kl", np.mean(bc_kls))
        self.logger.record("behavior_cloning/embed_std", np.mean(bc_embed_stds))
        self.logger.record("behavior_cloning/embed_dist", np.mean(bc_embed_dists))