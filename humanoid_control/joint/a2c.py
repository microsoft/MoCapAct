from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
from gym import spaces
from torch.nn import functional as F

from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

class A2CBC(A2C):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 0.0007,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1,
        kl_coef: float = 0,
        vf_coef: float = 0.5,
        bc_coef: float = 0.,
        max_grad_norm: float = 0.5,
        bc_dataloader: Optional[torch.utils.data.DataLoader] = None,
        rms_prop_eps: float = 0.00001,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            n_steps,
            gamma,
            gae_lambda,
            0.,
            vf_coef,
            max_grad_norm,
            rms_prop_eps,
            use_rms_prop,
            use_sde,
            sde_sample_freq,
            normalize_advantage,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model
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
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):

            actions = rollout_data.actions

            values, log_prob, kl_div = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values)
            
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
            else:
                bc_loss = 0.

            loss = policy_loss + self.kl_coef * torch.mean(kl_div) + self.vf_coef * value_loss + self.bc_coef * bc_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/kl", torch.mean(kl_div).item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())
            
        self.logger.record("behavior_cloning/loss", bc_loss.item())
        self.logger.record("behavior_cloning/mse", mse_bc.item())
        self.logger.record("behavior_cloning/kl", kl_bc.item())
        self.logger.record("behavior_cloning/embed_std", embed_std_bc.item())
        self.logger.record("behavior_cloning/embed_dist", embed_dist_bc.item())
