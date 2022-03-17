from abc import abstractmethod
import json
from typing import Any, Dict, Optional, Sequence, Text, Tuple, Type, Union
from functools import partial
import gym
import math
import numpy as np
from numpy.lib.arraysetops import isin
from stable_baselines3.common import policies
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.distributions.kl import kl_divergence
import pytorch_lightning as pl
from collections import deque

from stable_baselines3.common import torch_layers
from stable_baselines3.common.type_aliases import Schedule

from humanoid_control import utils
from humanoid_control.sb3 import features_extractor

def count_parameters(parameters):
    return sum(p.numel() for p in parameters)

class HierarchicalPolicy(policies.ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        ref_steps: Sequence[int],
        observables: Tuple[Sequence[Text]],
        embed_size: int = 60,
        ref_encoder_n_layers: int = 2,
        ref_encoder_layer_size: int = 1024,
        stand_encoder_n_layers: int = 2,
        stand_encoder_layer_size: int = 1024,
        decoder_n_layers: int = 3,
        decoder_layer_size: int = 1024,
        stand_vf_n_layers: int = 3,
        stand_vf_layer_size: int = 1024,
        layer_norm: bool = False,
        embedding_kl_weight: float = 0.1,
        embedding_correlation: float = 0.95,
        seq_steps: int = 30,
        activation_fn: Text = 'torch.nn.Tanh',
        squash_output: bool = False,
        std_dev: float = 0.1,
        ortho_init: bool = True,
        features_extractor_class: Type[features_extractor.CmuHumanoidFeaturesExtractor] = features_extractor.CmuHumanoidFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **ignored
    ):
        assert 0 <= embedding_correlation <= 1
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs['eps'] = 1e-5

        #features_extractor_kwargs['observable_keys'] = observables

        policies.BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output
        )

        self.activation_fn = utils.str_to_callable(activation_fn)
        self.observables = observables
        self.ref_steps = ref_steps
        self.std_dev = std_dev
        self.ortho_init = ortho_init
        self.embed_size = embed_size
        self.ref_encoder_n_layers = ref_encoder_n_layers
        self.ref_encoder_layer_size = ref_encoder_layer_size
        self.decoder_n_layers = decoder_n_layers
        self.decoder_layer_size = decoder_layer_size
        self.layer_norm = layer_norm
        self.embedding_kl_weight = embedding_kl_weight
        self.embedding_correlation = embedding_correlation
        self.embedding_std_dev = np.sqrt(1 - embedding_correlation**2)
        self.seq_steps = seq_steps

        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim
        
        reference_encoder_layers = torch_layers.create_mlp(
            self.features_extractor.sub_features_dim['ref_encoder'] + self.embed_size,
            2*self.embed_size,
            net_arch=self.ref_encoder_n_layers*[self.ref_encoder_layer_size],
            activation_fn=self.activation_fn
        )
        if self.layer_norm:
            for layer in range(1, 3*self.ref_encoder_n_layers-1, 3):
                reference_encoder_layers.insert(layer, nn.LayerNorm(self.ref_encoder_layer_size))
        self.reference_encoder = nn.Sequential(*reference_encoder_layers)

        action_decoder_layers = torch_layers.create_mlp(
            self.features_extractor.sub_features_dim['decoder'] + self.embed_size,
            self.action_space.shape[0]-self.embed_size,
            net_arch=self.decoder_n_layers*[self.decoder_layer_size],
            activation_fn=self.activation_fn,
            squash_output=self.squash_output
        )
        if self.layer_norm:
            for layer in range(1, 3*self.decoder_n_layers-1, 3):
                action_decoder_layers.insert(layer, nn.LayerNorm(self.decoder_layer_size))
        self.action_decoder = nn.Sequential(*action_decoder_layers)

        self.stand_encoder_n_layers = stand_encoder_n_layers
        self.stand_encoder_layer_size = stand_encoder_layer_size
        self.stand_vf_n_layers = stand_vf_n_layers
        self.stand_vf_layer_size = stand_vf_layer_size
        self.ortho_init = ortho_init

        stand_encoder_layers = torch_layers.create_mlp(
            self.features_extractor.sub_features_dim['stand_encoder'],
            2*self.embed_size,
            net_arch=self.stand_encoder_n_layers*[self.stand_encoder_layer_size],
            activation_fn=self.activation_fn
        )
        if self.layer_norm:
            for layer in range(1, 3*self.stand_encoder_n_layers-1, 3):
                stand_encoder_layers.insert(layer, nn.LayerNorm(self.stand_encoder_layer_size))
        self.stand_encoder = nn.Sequential(*stand_encoder_layers)

        stand_vf_layers = torch_layers.create_mlp(
            self.features_extractor.sub_features_dim['stand_encoder'],
            1,
            net_arch=self.stand_vf_n_layers*[self.stand_vf_layer_size],
            activation_fn=self.activation_fn
        )
        if self.layer_norm:
            for layer in range(1, 3*self.stand_vf_n_layers-1, 3):
                stand_vf_layers.insert(layer, nn.LayerNorm(self.stand_vf_layer_size))
        self.stand_vf = nn.Sequential(*stand_vf_layers)

        if self.ortho_init:
            self.stand_encoder.apply(partial(self.init_weights, gain=np.sqrt(2)))
            self.action_decoder.apply(partial(self.init_weights, gain=np.sqrt(2)))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = policies.BasePolicy._get_constructor_parameters(self)
        del data['normalize_images']
        data.update(
            dict(
                ref_steps=self.ref_steps,
                observables=self.observables,
                embed_size=self.embed_size,
                ref_encoder_n_layers=self.ref_encoder_n_layers,
                ref_encoder_layer_size=self.ref_encoder_layer_size,
                stand_encoder_n_layers=self.stand_encoder_n_layers,
                stand_encoder_layer_size=self.stand_encoder_layer_size,
                decoder_n_layers=self.decoder_n_layers,
                decoder_layer_size=self.decoder_layer_size,
                stand_vf_n_layers=self.stand_vf_n_layers,
                stand_vf_layer_size=self.stand_vf_layer_size,
                layer_norm=self.layer_norm,
                embedding_kl_weight=self.embedding_kl_weight,
                embedding_correlation=self.embedding_correlation,
                seq_steps=self.seq_steps,
                activation_fn=self.activation_fn,
                squash_output=self.squash_output,
                std_dev=self.std_dev,
                ortho_init=self.ortho_init,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs
            )
        )
        return data

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_features(observation)
        stand_obs, proprio = features['stand_encoder'], features['decoder']
        embed_gaussian = self._get_stand_encoder_distribution(stand_obs)
        embed = embed_gaussian.mean if deterministic else embed_gaussian.sample()
        act_gaussian = self._get_decoder_distribution(proprio, embed)
        return torch.cat([act_gaussian.mean, embed], dim=-1)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        stand_obs, proprio = features['stand_encoder'], features['decoder']
        embed_gaussian = self._get_stand_encoder_distribution(stand_obs)
        embed = embed_gaussian.mean if deterministic else embed_gaussian.sample()
        act_gaussian = self._get_decoder_distribution(proprio, embed)
        act = act_gaussian.mean if deterministic else act_gaussian.sample()

        actions = torch.cat([act, embed], dim=-1)
        values = self.stand_vf(stand_obs)
        log_prob = embed_gaussian.log_prob(embed) + act_gaussian.log_prob(act)

        return actions, values, log_prob

    def _get_ref_encoder_distribution(self, ref_obs: torch.Tensor, embed: torch.Tensor):
        embed_mean, embed_log_std = torch.split(
            self.reference_encoder(torch.cat([ref_obs, embed], dim=-1)),
            self.embed_size,
            dim=-1
        )
        return Independent(Normal(embed_mean, embed_log_std.exp()), -1)

    def _get_stand_encoder_distribution(self, stand_obs: torch.Tensor):
        embed_mean, embed_log_std = torch.split(
            self.stand_encoder(stand_obs),
            self.embed_size,
            dim=-1
        )
        return Independent(Normal(embed_mean, embed_log_std.exp()), -1)

    def _get_decoder_distribution(self, proprio: torch.Tensor, embed: torch.Tensor):
        act_mean = self.action_decoder(torch.cat([proprio, embed], dim=-1))
        return Independent(Normal(act_mean, self.std_dev), -1)
    
    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        stand_obs, proprio = features['stand_encoder'], features['decoder']
        act, embed = actions[..., :-self.embed_size], actions[..., -self.embed_size:]

        embed_gaussian = self._get_stand_encoder_distribution(stand_obs)
        act_gaussian = self._get_decoder_distribution(proprio, embed)

        values = self.stand_vf(stand_obs)
        log_prob = embed_gaussian.log_prob(embed) + act_gaussian.log_prob(act)
        prev_embed = obs['embedding']
        #prior = Independent(Normal(self.embedding_correlation*prev_embed, self.embedding_std_dev), -1)
        prior = Independent(Normal(torch.zeros_like(prev_embed), torch.ones_like(prev_embed)), -1)
        kl = kl_divergence(embed_gaussian, prior).mean()
        return values, log_prob, embed_gaussian

    def predict_values(self, obs):
        features = self.extract_features(obs)
        stand_obs = features['stand_encoder']
        return self.stand_vf(stand_obs)

    def compute_bc_loss(self, obs, actions, weights):
        #features = self.extract_features(obs)
        features = obs
        references, proprios = features['ref_encoder'], features['decoder']
        B, T, _ = actions.shape
        prev_embed = torch.randn(B, self.embed_size, dtype=actions.dtype, device=actions.device)
        total_log_prob, total_mse, total_kl, total_embed_std, total_embed_dist = 0, 0, 0, 0, 0
        for t in range(T):
            embed_gaussian = self._get_ref_encoder_distribution(references[:, t], prev_embed)
            embed = embed_gaussian.rsample()
            act_gaussian = self._get_decoder_distribution(proprios[:, t], embed)
            prior = Independent(Normal(self.embedding_correlation*prev_embed, self.embedding_std_dev), -1)

            log_prob = act_gaussian.log_prob(actions[:, t])
            mse = F.mse_loss(act_gaussian.mean, actions[:, t])
            kl = kl_divergence(embed_gaussian, prior).mean()
            total_log_prob += log_prob
            total_kl += kl
            total_mse += mse
            total_embed_std += torch.norm(embed_gaussian.stddev, dim=-1).mean().item() / np.sqrt(self.embed_size)
            total_embed_dist += torch.norm(embed_gaussian.mean - self.embedding_correlation*prev_embed, dim=-1).mean().item() / np.sqrt(self.embed_size)
            prev_embed = embed

        loss = (-weights@total_log_prob + self.embedding_kl_weight*total_kl) / T
        return loss, total_kl/T, total_mse/T, total_embed_std/T, total_embed_dist/T
