"""
Model definitions. Includes:
- MlpPolicy: A simple feedforward network.
- HierarchicalMlpPolicy: A feedforward network with a encoder and decoder architecture.
- NpmpPolicy: A recurrent stochastic encoder and feedforward decoder.
- McpPolicy: A recurrent stochastic encoder, gating layer for primitives, and low-level primitives.
- GPTPolicy: The GPT policy used for motion completion.

The models all inherit from the `BasePolicy` class, which itself is a subclass of
Stable-Baseline 3's `BasePolicy` and PyTorch Lightning's `LightningModule`.

The Stable-Baseline functionality is meant for efficient parallel evaluation (e.g., in the `evaluate.py` script).
The `_predict` function is to be implemented in the subclasses.

The PyTorch Lightning functionality is meant for efficient training.
The `initial_state`, `forward`, and `training_step` is to be implemented in the subclasses.
"""
from abc import abstractmethod
import json
from typing import Any, Dict, Optional, Sequence, Text, Tuple, Type, Union
import gym
import math
import numpy as np
from stable_baselines3.common import policies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.distributions.kl import kl_divergence
import pytorch_lightning as pl

from stable_baselines3.common import torch_layers

from mocapact import utils
from mocapact.sb3 import features_extractor
from mocapact.sb3.torch_layers import create_mlp

class BasePolicy(policies.BasePolicy, pl.LightningModule):
    """
    The base policy from which all other policy classes are derived.
    Inherits from both Stable Baseline's base policy class and PyTorch Lightning.
    The PyTorch functionality is used for training the policy from expert data.
    The Stable Baselines functionality is used for rolling out the trained
    policy in the environment, such as for evaluation.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        observables: Union[Sequence[Text], Dict[Text, Sequence[Text]]],
        ref_steps: Sequence[int],
        learning_rate: float,
        activation_fn: Text = 'torch.nn.Tanh',
        squash_output: bool = False,
        std_dev: float = 0.1,
        features_extractor_class: Type[torch_layers.BaseFeaturesExtractor] = features_extractor.CmuHumanoidFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_scheduler_class: Type[torch.optim.lr_scheduler._LRScheduler] = None,
        optimizer_scheduler_kwargs: Optional[Dict[str, Any]] = None
    ):
        policies.BasePolicy.__init__(self, observation_space, action_space,
                         features_extractor_class, features_extractor_kwargs,
                         None, False, optimizer_class, optimizer_kwargs,
                         squash_output=squash_output)
        pl.LightningModule.__init__(self)
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.features_extractor_kwargs['observable_keys'] = observables
        self.features_extractor = self.features_extractor_class(
            self.observation_space,
            **self.features_extractor_kwargs
        )
        self.optimizer_scheduler_class = optimizer_scheduler_class
        self.optimizer_scheduler_kwargs = optimizer_scheduler_kwargs
        self.observables = observables
        self.ref_steps = ref_steps
        self.std_dev = std_dev
        self.features_dim = self.features_extractor.features_dim
        self.activation_fn = utils.str_to_callable(activation_fn)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        del data['normalize_images']

        data.update(
            dict(
                observables=self.observables,
                ref_steps=self.ref_steps,
                learning_rate=self.learning_rate,
                activation_fn=self.activation_fn,
                squash_output=self.squash_output,
                std_dev=self.std_dev,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs
            )
        )
        return data

    ###########################
    # PyTorch Lightning methods
    ###########################
    @abstractmethod
    def initial_state(self, batch_size=1, deterministic=False):
        """
        For recurrent policies or observations with history (e.g., for GPT),
        the initial state for a training step or rollout episode.
        """
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        The function called for the policy training. The observation input should
        correspond to the one from the expert dataset after the feature extraction.
        """
        pass

    @abstractmethod
    def training_step(self, *args, **kwargs):
        """
        The function used within the training loop of PyTorch Lightning.
        Forwards the batch through the model and returns the supervision loss.
        Also logs any metrics of interest.
        """
        pass

    def configure_optimizers(self):
        self.optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate,
                                              **self.optimizer_kwargs)
        if self.optimizer_scheduler_class is not None:
            scheduler = dict(
                scheduler=self.optimizer_scheduler_class(self.optimizer, **self.optimizer_scheduler_kwargs),
                name='lr_scheduler'
            )
            return [self.optimizer], [scheduler]
        return self.optimizer

    #########################
    # Stable Baseline methods
    #########################
    @abstractmethod
    def _predict(
        self,
        observation: torch.Tensor,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The prediction function to be implemented.

        From Stable-Baseline's documentation:

        Get the action according to the policy for a given observation.

        :param observation: the pre-processed observation
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        pass

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[np.ndarray] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        From Stable-Baseline's documentation:

        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g., normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        obs = observation if isinstance(observation, torch.Tensor) else list(observation.values())[0]
        B = obs.shape[0]
        if episode_start is None:
            episode_start = np.array([False for _ in range(B)])
        init_state = self.initial_state(B, deterministic=deterministic)
        if state is None:
            state = init_state
        else:
            state[episode_start] = init_state[episode_start]
        state = torch.as_tensor(state).to(self.device)

        with torch.no_grad():
            actions, state = self._predict(observation, state, deterministic=deterministic)

        # Convert to numpy
        actions = actions.cpu().numpy()
        state = state.cpu().numpy()

        if self.squash_output:
            # Rescale to proper domain when using squashing
            actions = self.unscale_action(actions)
        else:
            # Actions could be on arbitrary scale, so clip the actions to avoid out of
            # bound error (e.g., if sampling from a Gaussian distribution)
            actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions[0]

        return actions, state


class MlpPolicy(BasePolicy):
    """
    The simple MLP.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        observables: Union[Sequence[Text], Tuple[Sequence[Text]]],
        ref_steps: Sequence[int],
        learning_rate: float,
        n_layers: int = 3,
        layer_size: int = 1024,
        activation_fn: Text = 'torch.nn.Tanh',
        squash_output: bool = False,
        std_dev: float = 0.1,
        features_extractor_class: Type[torch_layers.BaseFeaturesExtractor] = features_extractor.CmuHumanoidFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_scheduler_class: Type[torch.optim.lr_scheduler._LRScheduler] = None,
        optimizer_scheduler_kwargs: Optional[Dict[str, Any]] = None

    ):
        super().__init__(observation_space, action_space, observables, ref_steps, learning_rate,
                         activation_fn, squash_output, std_dev, features_extractor_class,
                         features_extractor_kwargs, optimizer_class, optimizer_kwargs,
                         optimizer_scheduler_class, optimizer_scheduler_kwargs)
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.mlp = nn.Sequential(*torch_layers.create_mlp(
            self.features_dim,
            self.action_space.shape[0],
            net_arch=self.n_layers*[self.layer_size],
            activation_fn=self.activation_fn,
            squash_output=self.squash_output
        ))

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                n_layers=self.n_layers,
                layer_size=self.layer_size
            )
        )
        return data

    def initial_state(self, batch_size=1, deterministic=False):
        return np.zeros(batch_size, dtype=np.float32)

    def forward(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        return Independent(Normal(self.mlp(features), self.std_dev), -1),

    def training_step(self, batch, batch_idx):
        obs, act, weights = batch
        features = self.extract_features(obs)
        act_gaussian, = self(features)
        loss = -weights @ act_gaussian.log_prob(act)
        mse = F.mse_loss(act, act_gaussian.mean)
        self.log("mse", mse, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def _predict(
        self,
        observation: torch.Tensor,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(observation)
        act_gaussian, = self(features)
        return (act_gaussian.mean, state)

class HierarchicalMlpPolicy(BasePolicy):
    """
    TODO: Use StochasticEncoder for high-level encoder.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        observables: Tuple[Sequence[Text]],
        ref_steps: Sequence[int],
        learning_rate: float,
        embed_size: int = 60,
        ref_encoder_n_layers: int = 2,
        ref_encoder_layer_size: int = 1024,
        decoder_n_layers: int = 3,
        decoder_layer_size: int = 1024,
        embedding_kl_coef: float = 0.1,
        activation_fn: Text = 'torch.nn.Tanh',
        squash_output: bool = False,
        std_dev: float = 0.1,
        features_extractor_class: Type[features_extractor.CmuHumanoidFeaturesExtractor] = features_extractor.CmuHumanoidFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_scheduler_class: Type[torch.optim.lr_scheduler._LRScheduler] = None,
        optimizer_scheduler_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__(observation_space, action_space, observables, ref_steps, learning_rate, activation_fn,
                         squash_output, std_dev, features_extractor_class, features_extractor_kwargs,
                         optimizer_class, optimizer_kwargs, optimizer_scheduler_class,
                         optimizer_scheduler_kwargs)
        self.embed_size = embed_size
        self.ref_encoder_n_layers = ref_encoder_n_layers
        self.ref_encoder_layer_size = ref_encoder_layer_size
        self.decoder_n_layers = decoder_n_layers
        self.decoder_layer_size = decoder_layer_size
        self.embedding_kl_coef = embedding_kl_coef

        self.reference_encoder = nn.Sequential(*torch_layers.create_mlp(
            self.features_extractor.sub_features_dim['ref_encoder'],
            2*self.embed_size,
            net_arch=self.encoder_n_layers*[self.encoder_layer_size],
            activation_fn=self.activation_fn
        ))
        self.action_decoder = nn.Sequential(*torch_layers.create_mlp(
            self.features_extractor.sub_features_dim['decoder'] + self.embed_size,
            self.action_space.shape[0],
            net_arch=self.decoder_n_layers*[self.decoder_layer_size],
            activation_fn=self.activation_fn,
            squash_output=self.squash_output
        ))

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                embed_size=self.embed_size,
                encoder_n_layers=self.encoder_n_layers,
                encoder_layer_size=self.encoder_layer_size,
                decoder_n_layers=self.decoder_n_layers,
                decoder_layer_size=self.decoder_layer_size,
                kl_weight=self.kl_weight
            )
        )
        return data

    def initial_state(self, batch_size=1, deterministic=False):
        return np.zeros(batch_size, dtype=np.float32)

    def forward(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reference, proprio = features['ref_encoder'], features['decoder']
        embed_mean, embed_log_std = torch.split(
            self.reference_encoder(reference),
            self.embed_size,
            dim=-1
        )
        embed_gaussian = Independent(Normal(embed_mean, embed_log_std.exp()), -1)
        embed = embed_mean if deterministic else embed_gaussian.rsample()

        act = self.action_decoder(torch.cat([proprio, embed], dim=-1))
        act_gaussian = Independent(Normal(act, self.std_dev), -1)

        return act_gaussian, embed_gaussian, embed

    def training_step(self, batch, batch_idx):
        obs, act, weights = batch
        features = self.extract_features(obs)
        act_gaussian, embed_gaussian, _ = self(features)

        log_prob = act_gaussian.log_prob(act)
        mse = F.mse_loss(act_gaussian.mean, act)
        spherical_gaussian = Independent(Normal(torch.zeros_like(embed_gaussian.mean), torch.ones_like(embed_gaussian.mean)), -1)
        kl = kl_divergence(embed_gaussian, spherical_gaussian).mean()
        loss = -weights@log_prob + self.embedding_kl_coef*kl
        self.log("mse", mse, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("kl_div", kl, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def _predict(
        self,
        observation: torch.Tensor,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(observation)
        act_gaussian, *_ = self.forward(features, deterministic=deterministic)
        return act_gaussian.mean, state

#######################################
# Policies that use stochastic encoder
#######################################

class StochasticEncoder(nn.Module):
    """
    Outputs a Gaussian distribution over encodings.
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        n_layers: int,
        layer_size: int,
        activation_fn: Type[nn.Module],
        layer_norm: bool = False,
        recurrent: bool = True,
        predict_std: bool = True,
        log_std_init: float = 0,
        min_std: float = 1e-5
    ):
        super().__init__()
        layers = create_mlp(
            input_dim+embed_dim if recurrent else input_dim,
            2*embed_dim if predict_std else embed_dim,
            net_arch=n_layers*[layer_size],
            activation_fn=activation_fn,
            layer_norm=layer_norm
        )
        if not predict_std:
            self.log_std = nn.Parameter(torch.ones(self.embed_dim)*log_std_init, requires_grad=True)
        self.encoder = nn.Sequential(*layers)
        self.recurrent = recurrent
        self.predict_std = predict_std
        self.embed_dim = embed_dim
        self.log_std_init = log_std_init
        self.min_std = min_std

    def forward(self, input: torch.Tensor, prev_embed: torch.Tensor = None):
        if self.recurrent:
            assert prev_embed is not None
        inp = torch.cat([input, prev_embed], dim=-1) if self.recurrent else input
        if self.predict_std:
            embed_mean, embed_log_std = torch.split(
                self.encoder(inp),
                self.embed_dim,
                dim=-1
            )
        else:
            embed_mean = self.encoder(inp)
            embed_log_std = self.log_std
        std = torch.clamp(embed_log_std.exp(), min=self.min_std)
        return Independent(Normal(embed_mean, std), 1)

class NpmpPolicy(BasePolicy):
    """
    'Neural probabilistic motor primitive' architecture originally proposed in [1]
    and the multi-clip policy used in MoCapAct. Consists of a recurrent stochastic encoder
    and a feedforward decoder.

    [1] Merel, Josh, Leonard Hasenclever, Alexandre Galashov, Arun Ahuja, Vu Pham,
        Greg Wayne, Yee Whye Teh, and Nicolas Heess. "Neural probabilistic motor
        primitives for humanoid control." arXiv preprint arXiv:1811.11711 (2018).
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        observables: Dict[Text, Sequence[Text]],
        ref_steps: Sequence[int],
        learning_rate: float,
        embed_size: int = 60,
        ref_encoder_n_layers: int = 2,
        ref_encoder_layer_size: int = 1024,
        decoder_n_layers: int = 3,
        decoder_layer_size: int = 1024,
        recurrent_encoder: bool = True,
        layer_norm: bool = False,
        embedding_kl_weight: float = 0.1,
        embedding_correlation: float = 0.95,
        seq_steps: int = 30,
        activation_fn: Text = 'torch.nn.Tanh',
        squash_output: bool = False,
        std_dev: float = 0.1,
        features_extractor_class: Type[features_extractor.CmuHumanoidFeaturesExtractor] = features_extractor.CmuHumanoidFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_scheduler_class: Type[torch.optim.lr_scheduler._LRScheduler] = None,
        optimizer_scheduler_kwargs: Optional[Dict[str, Any]] = None
    ):
        assert 0 <= embedding_correlation < 1
        super().__init__(observation_space, action_space, observables, ref_steps, learning_rate, activation_fn,
                         squash_output, std_dev, features_extractor_class, features_extractor_kwargs,
                         optimizer_class, optimizer_kwargs, optimizer_scheduler_class,
                         optimizer_scheduler_kwargs)
        self.embed_size = embed_size
        self.ref_encoder_n_layers = ref_encoder_n_layers
        self.ref_encoder_layer_size = ref_encoder_layer_size
        self.decoder_n_layers = decoder_n_layers
        self.decoder_layer_size = decoder_layer_size
        self.recurrent_encoder = recurrent_encoder
        self.layer_norm = layer_norm
        self.embedding_kl_weight = embedding_kl_weight
        self.embedding_correlation = embedding_correlation if self.recurrent_encoder else 0.
        self.embedding_std_dev = np.sqrt(1 - embedding_correlation**2)
        self.seq_steps = seq_steps

        self.reference_encoder = StochasticEncoder(
            self.features_extractor.sub_features_dim['ref_encoder'],
            self.embed_size,
            self.ref_encoder_n_layers,
            self.ref_encoder_layer_size,
            self.activation_fn,
            layer_norm=self.layer_norm,
            recurrent=self.recurrent_encoder,
            predict_std=True
        )

        action_decoder_layers = create_mlp(
            self.features_extractor.sub_features_dim['decoder'] + self.embed_size,
            self.action_space.shape[0],
            net_arch=self.decoder_n_layers*[self.decoder_layer_size],
            activation_fn=self.activation_fn,
            squash_output=self.squash_output,
            layer_norm=self.layer_norm
        )
        self.action_decoder = nn.Sequential(*action_decoder_layers)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                embed_size=self.embed_size,
                ref_encoder_n_layers=self.ref_encoder_n_layers,
                ref_encoder_layer_size=self.ref_encoder_layer_size,
                decoder_n_layers=self.decoder_n_layers,
                decoder_layer_size=self.decoder_layer_size,
                recurrent_encoder=self.recurrent_encoder,
                layer_norm=self.layer_norm,
                embedding_kl_weight=self.embedding_kl_weight,
                embedding_correlation=self.embedding_correlation,
                seq_steps=self.seq_steps,
                truncated_bptt_steps=self.truncated_bptt_steps
            )
        )
        return data

    def initial_state(self, batch_size=1, deterministic=False):
        if deterministic:
            return np.zeros((batch_size, self.embed_size)).astype(np.float32)
        return np.random.randn(batch_size, self.embed_size).astype(np.float32)

    def forward(
        self,
        ref_encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        prev_embed: torch.Tensor,
        deterministic: bool = False
    ):
        embed, embed_distribution = self.ref_encoder_forward(ref_encoder_input, prev_embed, deterministic)
        act_distribution = self.action_decoder_forward(decoder_input, embed)
        return act_distribution, embed_distribution, embed

    def ref_encoder_forward(
        self,
        ref_encoder_input: torch.Tensor,
        prev_embed: torch.Tensor,
        deterministic: bool = False
    ):
        embed_distribution = self.reference_encoder(ref_encoder_input, prev_embed)
        embed = embed_distribution.mean if deterministic else embed_distribution.rsample()

        return embed, embed_distribution

    def action_decoder_forward(
        self,
        proprio_input: torch.Tensor,
        embed: torch.Tensor
    ):
        act = self.action_decoder(torch.cat([proprio_input, embed], dim=-1))
        act_distribution = Independent(Normal(act, self.std_dev), 1)

        return act_distribution

    def low_level_policy(self, observation, embed):
        """
        To be used in RL transfer tasks.
        """
        #features = self.extract_features(observation)
        features = self.features_extractor(observation)
        proprio = features['decoder']
        act_distribution = self.action_decoder_forward(proprio, embed)
        return act_distribution.mean

    def run_batch(self, batch):
        obs, acts, weights = batch
        features = self.extract_features(obs)
        references, proprios = features['ref_encoder'], features['decoder']
        B, T, _ = acts.shape
        embed = torch.as_tensor(self.initial_state(B)).type_as(acts)
        total_kl, total_embed_std, total_delta_embed = 0, 0, 0
        all_embeds = [embed]
        for t in range(T):
            next_embed, next_embed_distribution = self.ref_encoder_forward(references[:, t], embed)
            prior_embed_distribution = Independent(Normal(self.embedding_correlation*embed, self.embedding_std_dev), 1)
            kl = kl_divergence(next_embed_distribution, prior_embed_distribution).sum()
            total_kl += kl
            with torch.no_grad():
                total_embed_std += next_embed_distribution.stddev.mean()
                total_delta_embed += torch.mean(torch.abs(next_embed_distribution.mean - prior_embed_distribution.mean))

            embed = next_embed
            all_embeds.append(embed)
        embeds = torch.stack(all_embeds, 1)

        act_distribution = self.action_decoder_forward(proprios, embeds[:, 1:])
        log_prob = act_distribution.log_prob(acts)
        weighted_log_prob = torch.einsum('ij,ij', weights, log_prob)
        loss = (-weighted_log_prob + self.embedding_kl_weight*total_kl) / (B*T)

        with torch.no_grad():
            mse = F.mse_loss(act_distribution.mean, acts)
            embed_mean = torch.mean(embeds)
            embed_std = torch.std(embeds)

        return loss, weighted_log_prob/(B*T), total_kl/(B*T), mse, total_delta_embed/T, total_embed_std/T, embed_mean, embed_std

    def training_step(self, batch, batch_idx):
        loss, weighted_log_prob, kl, mse, delta_embed_mean, delta_embed_std, embed_mean, embed_std = self.run_batch(batch)
        log = lambda name, metric: self.log(name, metric, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        log("loss/loss", loss.item())
        log("loss/weighted_log_prob_loss", -weighted_log_prob.item())
        log("loss/mse", mse.item())
        log("loss/kl_div", kl.item())
        log("embed/delta_mean", delta_embed_mean.item())
        log("embed/delta_std", delta_embed_std.item())
        log("embed/embed_mean", embed_mean.item())
        log("embed/embed_std", embed_std.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, weighted_log_prob, kl, mse, _, _, _, _ = self.run_batch(batch)

        log = lambda name, metric: self.log(name, metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        log("val_loss/loss", loss.item())
        log("val_loss/weighted_log_prob_loss", -weighted_log_prob.item())
        log("val_loss/mse", mse.item())
        log("val_loss/kl_div", kl.item())
        return loss

    def _predict(
        self,
        observation: torch.Tensor,
        embed: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(observation)
        references, proprios = features['ref_encoder'], features['decoder']
        act_distribution, _, next_embed = self.forward(references, proprios, embed, deterministic=deterministic)
        return act_distribution.mean, next_embed

class McpPrimitives(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_shared_layers: int,
        shared_layer_size: int,
        n_primitives: int,
        n_primitive_layers: int,
        primitive_layer_size: int,
        activation_fn: Type[nn.Module],
        layer_norm: bool = False
    ):
        super().__init__()
        shared_layers = create_mlp(
            obs_dim,
            shared_layer_size,
            net_arch=(n_shared_layers-1)*[shared_layer_size],
            activation_fn=activation_fn,
            layer_norm=layer_norm
        )
        if layer_norm:
            shared_layers.append(nn.LayerNorm(shared_layer_size))
        shared_layers.append(activation_fn())
        self.shared_net = nn.Sequential(*shared_layers)

        self.n_primitives = n_primitives
        self.act_dim = act_dim
        primitives = []
        for _ in range(n_primitives):
            primitive_layers = create_mlp(
                shared_layer_size,
                2*act_dim,
                net_arch=n_primitive_layers*[primitive_layer_size],
                activation_fn=activation_fn,
                layer_norm=layer_norm
            )
            primitives.append(nn.Sequential(*primitive_layers))
        self.primitive_nets = nn.ModuleList(primitives)

    def forward(self, obs: torch.Tensor):
        shared_features = self.shared_net(obs)
        means, log_stds = [], []
        for primitive_net in self.primitive_nets:
            mean, log_std = torch.split(
                primitive_net(shared_features),
                self.act_dim,
                dim=-1
            )
            means.append(mean)
            log_stds.append(log_std)
        return torch.stack(means, dim=-2), torch.stack(log_stds, dim=-2)

class McpPolicy(BasePolicy):
    """
    The 'multiplicative compositional policy' architecture [1]. Our implementation consists
    of a recurrent stochastic encoder, gating layer for choosing the primitives, and
    low-level primitives layer.

    [1] Peng, Xue Bin, Michael Chang, Grace Zhang, Pieter Abbeel, and Sergey Levine.
        "MCP: Learning composable hierarchical control with multiplicative compositional policies."
        Advances in Neural Information Processing Systems 32 (2019).
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        observables: Dict[Text, Sequence[Text]],
        ref_steps: Sequence[int],
        learning_rate: float,
        embed_size: int = 60,
        ref_encoder_n_layers: int = 2,
        ref_encoder_layer_size: int = 1024,
        n_primitives: int = 8,
        gating_n_layers: int = 2,
        gating_layer_size: int = 512,
        n_shared_primitive_layers: int = 2,
        shared_primitive_layer_size: int = 512,
        n_primitive_layers: int = 1,
        primitive_layer_size: int = 512,
        layer_norm: bool = True,
        min_primitive_std: float = 0.01,
        embedding_kl_weight: float = 0.1,
        embedding_correlation: float = 0.95,
        seq_steps: int = 30,
        activation_fn: Text = 'torch.nn.ELU',
        squash_output: bool = False,
        features_extractor_class: Type[features_extractor.CmuHumanoidFeaturesExtractor] = features_extractor.CmuHumanoidFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_scheduler_class: Type[torch.optim.lr_scheduler._LRScheduler] = None,
        optimizer_scheduler_kwargs: Optional[Dict[str, Any]] = None
    ):
        assert 0 <= embedding_correlation <= 1
        super().__init__(observation_space, action_space, observables, ref_steps, learning_rate, activation_fn,
                         squash_output, 0.1, features_extractor_class, features_extractor_kwargs,
                         optimizer_class, optimizer_kwargs, optimizer_scheduler_class,
                         optimizer_scheduler_kwargs)
        self.embed_size = embed_size
        self.ref_encoder_n_layers = ref_encoder_n_layers
        self.ref_encoder_layer_size = ref_encoder_layer_size
        self.n_primitives = n_primitives
        self.gating_n_layers = gating_n_layers
        self.gating_layer_size = gating_layer_size
        self.n_shared_primitive_layers = n_shared_primitive_layers
        self.shared_primitive_layer_size = shared_primitive_layer_size
        self.n_primitive_layers = n_primitive_layers
        self.primitive_layer_size = primitive_layer_size
        self.layer_norm = layer_norm
        self.min_primtive_std = min_primitive_std
        self.embedding_kl_weight = embedding_kl_weight
        self.embedding_correlation = embedding_correlation
        self.embedding_std_dev = np.sqrt(1 - embedding_correlation**2)
        self.seq_steps = seq_steps

        self.reference_encoder = StochasticEncoder(
            self.features_extractor.sub_features_dim['ref_encoder'],
            self.embed_size,
            self.ref_encoder_n_layers,
            self.ref_encoder_layer_size,
            self.activation_fn,
            self.layer_norm
        )

        gating_layers = create_mlp(
            self.features_extractor.sub_features_dim['decoder'] + self.embed_size,
            self.n_primitives,
            net_arch=self.gating_n_layers*[self.gating_layer_size],
            activation_fn=self.activation_fn,
            layer_norm=self.layer_norm
        )
        #gating_layers.append(nn.LogSigmoid())
        self.gating_net = nn.Sequential(*gating_layers)

        self.primitives = McpPrimitives(
            self.features_extractor.sub_features_dim['decoder'],
            self.action_space.shape[0],
            self.n_shared_primitive_layers,
            self.shared_primitive_layer_size,
            self.n_primitives,
            self.n_primitive_layers,
            self.primitive_layer_size,
            self.activation_fn,
            layer_norm=self.layer_norm
        )

    def initial_state(self, batch_size=1, deterministic=False):
        if deterministic:
            return np.zeros((batch_size, self.embed_size)).astype(np.float32)
        return np.random.randn(batch_size, self.embed_size).astype(np.float32)

    def forward(
        self,
        ref_encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        prev_embed: torch.Tensor,
        deterministic: bool = False
    ):
        embed, embed_distribution = self.ref_encoder_forward(ref_encoder_input, prev_embed, deterministic)
        act_distribution, _ = self.action_decoder_forward(decoder_input, embed)
        return act_distribution, embed_distribution, embed

    def ref_encoder_forward(
        self,
        ref_encoder_input: torch.Tensor,
        prev_embed: torch.Tensor,
        deterministic: bool = False
    ):
        embed_distribution = self.reference_encoder(ref_encoder_input, prev_embed)
        embed = embed_distribution.mean if deterministic else embed_distribution.rsample()

        return embed, embed_distribution

    def action_decoder_forward(
        self,
        proprio_input: torch.Tensor,
        embed: torch.Tensor
    ):
        log_weights = self.gating_net(torch.cat([proprio_input, embed], dim=-1))
        means, log_stds = self.primitives(proprio_input)
        log_stds = log_stds.clamp(min=np.log(self.min_primtive_std))

        scores = log_weights.unsqueeze(-1) - log_stds
        primitive_weights = torch.softmax(scores, dim=-2)
        act = torch.sum(primitive_weights * means, dim=-2)
        log_std = -torch.logsumexp(scores, dim=-2)
        act_distribution = Independent(Normal(act, log_std.exp()), 1)

        return act_distribution, (primitive_weights,)

    def training_step(self, batch, batch_idx, hiddens):
        obs, acts, weights = batch
        features = self.extract_features(obs)
        references, proprios = features['ref_encoder'], features['decoder']
        B, T, _ = acts.shape
        embed = hiddens if hiddens is not None else torch.as_tensor(self.initial_state(B)).type_as(acts)
        total_kl, total_embed_std, total_delta_embed = 0, 0, 0
        all_embeds = [embed]
        for t in range(T):
            next_embed, next_embed_distribution = self.ref_encoder_forward(references[:, t], embed)
            prior_embed_distribution = Independent(Normal(self.embedding_correlation*embed, self.embedding_std_dev), 1)
            kl = kl_divergence(next_embed_distribution, prior_embed_distribution).sum()
            total_kl += kl
            with torch.no_grad():
                total_embed_std += next_embed_distribution.stddev.mean()
                total_delta_embed += torch.mean(torch.abs(next_embed_distribution.mean - prior_embed_distribution.mean))

            embed = next_embed
            all_embeds.append(embed)
        embeds = torch.stack(all_embeds, 1)

        act_distribution, (primitive_weights,) = self.action_decoder_forward(proprios, embeds[:, 1:])
        log_prob = act_distribution.log_prob(acts)
        weighted_log_prob = torch.einsum('ij,ij', weights, log_prob)
        loss = (-weighted_log_prob + self.embedding_kl_weight*total_kl) / (B*T)

        with torch.no_grad():
            mse = F.mse_loss(act_distribution.mean, acts)
            embed_mean = torch.mean(embeds)
            embed_std = torch.std(embeds)
            primitive_weights_entropy = torch.mean(torch.sum(-primitive_weights * primitive_weights.log(), dim=-2)) / np.log(self.n_primitives)
            act_entropy = act_distribution.entropy().mean()

        log = lambda name, metric: self.log(name, metric, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        log("loss/mse", mse.item())
        log("loss/weighted_log_prob_loss", -weighted_log_prob.item() / (B*T))
        log("loss/kl_div", total_kl.item()/ (B*T))
        log("loss/loss", loss.item())
        log("embed/delta_mean", total_delta_embed.item()/T)
        log("embed/delta_std", total_embed_std.item()/T)
        log("embed/embed_mean", embed_mean.item())
        log("embed/embed_std", embed_std.item())
        log("mcp/primitive_weights_entropy", primitive_weights_entropy.item())
        log("mcp/act_entropy", act_entropy.item())
        return dict(loss=loss, hiddens=embed)

    def _predict(
        self,
        observation: torch.Tensor,
        embed: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(observation)
        references, proprios = features['ref_encoder'], features['decoder']
        act_distribution, _, next_embed = self.forward(references, proprios, embed, deterministic=deterministic)
        return act_distribution.mean, next_embed

    def tbptt_split_batch(self, batch, split_size):
        obs, acts, weights = batch
        T = acts.shape[1]
        splits = []
        for t in range(0, T, split_size):
            obs_subseq = {k: v[:, t:t+split_size] for k, v in obs.items()}
            acts_subseq = acts[:, t:t+split_size]
            weights_subseq = weights[:, t:t+split_size]
            splits.append((obs_subseq, acts_subseq, weights_subseq))
        return splits


#######
# GPT
#######
class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, obs_size, action_size, block_size, **kwargs):
        self.obs_size = obs_size
        self.action_size = action_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

    def to_json(self, output_fname):
        with open(output_fname, 'w') as f:
            f.write(json.dumps(self.__dict__))

    @staticmethod
    def from_json(fname):
        with open(fname, 'r') as f:
            kwargs = json.loads(f.read())
            return GPTConfig(**kwargs)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Linear(config.obs_size, config.n_embd)
        # self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.action_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, learning_rate, weight_decay, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        allowlist_weight_modules = (torch.nn.Linear, )
        blocklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, allowlist_weight_modules):
                    # weights of allowed modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blocklist_weight_modules):
                    # weights of blocked modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(self, idx):
        added_dim = False
        if idx.ndim == 2:
            idx = idx.unsqueeze(1) # Hack to add singular time dimension
            added_dim = True
        b, t, d = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        x = self.head(x)

        if added_dim:
            x = x.squeeze(1) # Remove the time dim if we added it
        return x

class GPTPolicy(BasePolicy):
    """
    The GPT policy used for motion completion.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        observables: Union[Sequence[Text], Dict[Text, Sequence[Text]]],
        ref_steps: Sequence[int],
        learning_rate: float = 0.000003,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.95),
        block_size: int = 16,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 768,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        std_dev: float = 0.1,
        features_extractor_class: Type[features_extractor.CmuHumanoidFeaturesExtractor] = features_extractor.CmuHumanoidFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_scheduler_class: Type[torch.optim.lr_scheduler._LRScheduler] = None,
        optimizer_scheduler_kwargs: Optional[Dict[str, Any]] = None

    ):
        super().__init__(observation_space, action_space, observables, ref_steps, learning_rate,
                         'torch.nn.Tanh', False, std_dev, features_extractor_class,
                         features_extractor_kwargs, optimizer_class, optimizer_kwargs,
                         optimizer_scheduler_class, optimizer_scheduler_kwargs)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.block_size = block_size
        self.gpt = GPT(
            GPTConfig(
                obs_size=self.features_dim,
                action_size=action_space.shape[0],
                block_size=block_size,
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
                embd_pdrop=embd_pdrop,
                resid_pdrop=resid_pdrop,
                attn_pdrop=attn_pdrop,
            )
        )

    def initial_state(self, batch_size=1, deterministic=False):
        return np.array([], dtype=np.float32)

    def forward(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        return Independent(Normal(self.gpt(features), self.std_dev), -1),

    def training_step(self, batch, batch_idx):
        obs, act, weights = batch
        features = self.extract_features(obs)
        act_gaussian, = self(features)
        loss = -weights[:, -1] @ act_gaussian.log_prob(act)
        mse = F.mse_loss(act, act_gaussian.mean)
        self.log("loss/mse", mse, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("loss/loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        obs, act, weights = batch
        features = self.extract_features(obs)
        act_gaussian, = self(features)
        loss = -weights[:, -1] @ act_gaussian.log_prob(act)
        mse = F.mse_loss(act, act_gaussian.mean)
        self.log("val_loss/mse", mse, on_step=False, on_epoch=True, logger=True)
        self.log("val_loss/loss", loss, on_step=False, on_epoch=True, logger=True)
        return mse


    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        self.set_training_mode(not deterministic)

        observation, vectorized_env = self.obs_to_tensor(observation)

        obs = observation if isinstance(observation, torch.Tensor) else list(observation.values())[0]
        B = obs.shape[0]
        if episode_start is None:
            episode_start = np.array([False for _ in range(B)])
        if state is None:
            # Build an empty context, and set pointer to zero (beginning of context)
            context = np.zeros([B, self.block_size, self.features_extractor.features_dim], dtype=np.float32)
            state = (context, np.zeros(B, dtype=np.int64))
        else:
            # Set pointers corresponding to episode_start to zero
            state[1][episode_start] = 0
        state = [torch.as_tensor(x).to(self.device) for x in state]

        with torch.no_grad():
            actions, state = self._predict(observation, state, deterministic=deterministic)

        # Convert to numpy
        actions = actions.cpu().numpy()
        state = [x.cpu().numpy() for x in state]

        if self.squash_output:
            actions = self.unscale_action(actions)
        else:
            actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            actions = actions[0]

        return actions, state

    def _predict(
        self,
        observation: torch.Tensor,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(observation)

        # context is a queue, and idx tells us where to put next token
        context, idx = state

        # if we filled the queue, we'll pop the first token
        pop = (idx == self.block_size)
        idx = torch.clamp(idx, max=self.block_size-1)

        # pop the first token where needed
        context[pop] = torch.roll(context[pop], -1, 1)

        # put the token into the latest slot
        context[np.arange(len(idx)), idx] = features[np.arange(len(idx))]
        act_gaussian, = self.forward(context) # BxTxD
        act_seq = act_gaussian.mean

        # get the action from the proper timestep
        act = act_seq[np.arange(len(idx)), idx]
        idx += 1
        state = (context, idx)
        
        return act, state

    def configure_optimizers(self):
        return self.gpt.configure_optimizers(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
