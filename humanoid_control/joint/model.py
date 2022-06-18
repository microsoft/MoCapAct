from typing import Any, Dict, Optional, Sequence, Text, Tuple, Type
from functools import partial
import gym
import numpy as np
from stable_baselines3.common import policies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.distributions.kl import kl_divergence

from stable_baselines3.common.type_aliases import Schedule

from humanoid_control.distillation import model
from humanoid_control.sb3 import features_extractor
from humanoid_control.sb3 import torch_layers

def count_parameters(parameters):
    return sum(p.numel() for p in parameters)

class HierarchicalPolicy(policies.ActorCriticPolicy, model.NpmpPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        ref_steps: Sequence[int],
        observables: Dict[Text, Sequence[Text]],
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
        ref_embedding_kl_weight: float = 0.1,
        stand_embedding_kl_weight: float = 0.05,
        embedding_correlation: float = 0.95,
        predict_delta_embed: bool = False,
        bc_seq_steps: int = 30,
        bc_truncated_bptt_steps: Optional[int] = None,
        activation_fn: Text = 'torch.nn.Tanh',
        squash_output: bool = False,
        std_dev: float = 0.1,
        ortho_init: bool = True,
        log_embed_std_init: float = np.log(0.5),
        act_std_schedule: Schedule = None,
        features_extractor_class: Type[features_extractor.CmuHumanoidFeaturesExtractor] = features_extractor.CmuHumanoidFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **ignored
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs['eps'] = 1e-5

        #features_extractor_kwargs = features_extractor_kwargs or dict()
        #features_extractor_kwargs['observable_keys'] = observables

        model.NpmpPolicy.__init__(self, observation_space, action_space, observables, ref_steps, np.nan, embed_size,
                         ref_encoder_n_layers, ref_encoder_layer_size, decoder_n_layers, decoder_layer_size,
                         layer_norm, ref_embedding_kl_weight, embedding_correlation, predict_delta_embed,
                         bc_seq_steps, bc_truncated_bptt_steps, activation_fn, squash_output,
                         std_dev, features_extractor_class, features_extractor_kwargs, optimizer_class,
                         optimizer_kwargs)

        self.stand_encoder_n_layers = stand_encoder_n_layers
        self.stand_encoder_layer_size = stand_encoder_layer_size
        self.stand_vf_n_layers = stand_vf_n_layers
        self.stand_vf_layer_size = stand_vf_layer_size
        self.stand_embedding_kl_weight = stand_embedding_kl_weight
        self.ortho_init = ortho_init
        self.log_std_init = log_std_init
        self.log_embed_std_init = log_embed_std_init

        self.stand_encoder = model.RecurrentEncoder(
            self.features_extractor.sub_features_dim['stand_encoder']-self.embed_size,
            self.embed_size,
            self.stand_encoder_n_layers,
            self.stand_encoder_layer_size,
            self.activation_fn,
            self.layer_norm,
            self.predict_delta_embed,
            self.embedding_correlation
        )

        stand_vf_layers = torch_layers.create_mlp(
            self.features_extractor.sub_features_dim['stand_encoder'],
            1,
            net_arch=self.stand_vf_n_layers*[self.stand_vf_layer_size],
            activation_fn=self.activation_fn,
            layer_norm=self.layer_norm
        )
        self.stand_vf = nn.Sequential(*stand_vf_layers)

        # TODO: clean up hack of having to overwrite the decoder
        action_decoder_layers = torch_layers.create_mlp(
            self.features_extractor.sub_features_dim['decoder'] + self.embed_size,
            self.action_space.shape[0]-self.embed_size,
            net_arch=self.decoder_n_layers*[self.decoder_layer_size],
            activation_fn=self.activation_fn,
            squash_output=self.squash_output,
            layer_norm=self.layer_norm
        )
        self.action_decoder = nn.Sequential(*action_decoder_layers)


        if self.ortho_init:
            self.stand_encoder.apply(partial(self.init_weights, gain=np.sqrt(2)))
            self.action_decoder.apply(partial(self.init_weights, gain=np.sqrt(2)))

        #self.log_std = nn.Parameter(torch.ones(self.action_space.shape[0]-self.embed_size)*self.log_std_init, requires_grad=False)
        self.log_std = torch.ones(self.action_space.shape[0]-self.embed_size)*self.log_std_init
        self.log_embed_std = nn.Parameter(torch.ones(self.embed_size)*self.log_embed_std_init, requires_grad=True)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = model.NpmpPolicy._get_constructor_parameters(self)
        data.update(
            dict(
                stand_encoder_n_layers=self.stand_encoder_n_layers,
                stand_encoder_layer_size=self.stand_encoder_layer_size,
                stand_vf_n_layers=self.stand_vf_n_layers,
                stand_vf_layer_size=self.stand_vf_layer_size,
                stand_embedding_kl_weight=self.stand_embedding_kl_weight,
                log_std_init=self.log_std_init,
                log_embed_std_init=self.log_embed_std_init
            )
        )
        return data

    def _predict(self, observation: torch.Tensor, state, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_features(observation)
        stand_obs, proprio = features['stand_encoder'], features['decoder']
        embed_distribution = self.stand_encoder_forward(stand_obs, deterministic=deterministic)
        embed = embed_distribution.mean if deterministic else embed_distribution.sample()
        act_distribution = self.action_decoder_forward(proprio, embed)
        return torch.cat([act_distribution.mean, embed], dim=-1), state

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        stand_obs, proprio = features['stand_encoder'], features['decoder']
        embed_distribution = self.stand_encoder_forward(stand_obs)
        embed = embed_distribution.mean if deterministic else embed_distribution.sample()
        act_distribution = self.action_decoder_forward(proprio, embed)
        act_distribution = Independent(Normal(act_distribution.mean, self.log_std.exp()), 1)
        act = act_distribution.mean if deterministic else act_distribution.sample()

        actions = torch.cat([act, embed], dim=-1)
        values = self.stand_vf(stand_obs)
        log_prob = embed_distribution.log_prob(embed) + act_distribution.log_prob(act)

        return actions, values, log_prob

    def stand_encoder_forward(
        self,
        stand_encoder_input: torch.Tensor,
        deterministic: bool = False
    ):
        obs, prev_embed = torch.split(stand_encoder_input, stand_encoder_input.shape[-1]-self.embed_size, -1)
        embed_distribution = self.stand_encoder(obs, prev_embed)
        embed_distribution = Independent(Normal(embed_distribution.mean, self.log_embed_std.exp()), 1)
        #embed_distribution = Independent(Normal(self.embedding_correlation*prev_embed, self.embedding_std_dev), 1)
        return embed_distribution

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        stand_obs, proprio = features['stand_encoder'], features['decoder']
        act, embed = torch.split(actions, [actions.shape[-1]-self.embed_size, self.embed_size], -1)

        embed_distribution = self.stand_encoder_forward(stand_obs)
        act_distribution = self.action_decoder_forward(proprio, embed)
        act_distribution = Independent(Normal(act_distribution.mean, self.log_std.exp()), 1)

        values = self.stand_vf(stand_obs)
        log_prob = embed_distribution.log_prob(embed) + act_distribution.log_prob(act)
        return values, log_prob, embed_distribution

    def predict_values(self, obs):
        features = self.extract_features(obs)
        stand_obs = features['stand_encoder']
        return self.stand_vf(stand_obs)

    def compute_bc_loss(self, obs, actions, weights):
        #features = self.extract_features(obs)
        features = obs
        references, proprios = features['ref_encoder'], features['decoder']
        B, T, _ = actions.shape
        embed = torch.randn(B, self.embed_size, dtype=actions.dtype, device=actions.device)

        total_kl, total_embed_std, total_delta_embed = 0, 0, 0
        all_embeds = [embed]
        for t in range(T):
            next_embed, next_embed_distribution = self.ref_encoder_forward(references[:, t], embed)
            prior_embed_distribution = Independent(Normal(self.embedding_correlation*embed, self.embedding_std_dev), 1)
            kl = kl_divergence(next_embed_distribution, prior_embed_distribution).sum()
            total_kl += kl
            with torch.no_grad():
                total_embed_std += next_embed_distribution.stddev.mean()
                total_delta_embed += torch.mean(torch.abs(next_embed_distribution.mean - self.embedding_correlation*embed))

            embed = next_embed
            all_embeds.append(embed)
        embeds = torch.stack(all_embeds, 1)

        act_distribution = self.action_decoder_forward(proprios, embeds[:, 1:])
        log_prob = act_distribution.log_prob(actions)
        weighted_log_prob = torch.einsum('ij,ij', weights, log_prob)
        loss = (-weighted_log_prob + self.embedding_kl_weight*total_kl) / (B*T)

        with torch.no_grad():
            mse = F.mse_loss(act_distribution.mean, actions)
            embed_mean = torch.mean(embeds)
            embed_std = torch.std(embeds)
        return loss, -weighted_log_prob.item() / (B*T), total_kl/(B*T), mse.item(), total_delta_embed.item()/T, total_embed_std.item()/T, embed_mean.item(), embed_std.item()


class StandUpPolicy(policies.ActorCriticPolicy, model.NpmpPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        ref_steps: Sequence[int],
        observables: Dict[Text, Sequence[Text]],
        embed_size: int = 60,
        ref_encoder_n_layers: int = 2,
        ref_encoder_layer_size: int = 1024,
        decoder_n_layers: int = 3,
        decoder_layer_size: int = 1024,
        stand_vf_n_layers: int = 3,
        stand_vf_layer_size: int = 1024,
        layer_norm: bool = False,
        ref_embedding_kl_weight: float = 0.1,
        embedding_correlation: float = 0.95,
        bc_seq_steps: int = 30,
        activation_fn: Text = 'torch.nn.Tanh',
        squash_output: bool = False,
        std_dev: float = 0.1,
        ortho_init: bool = True,
        log_std_init: float = 0.,
        features_extractor_class: Type[features_extractor.CmuHumanoidFeaturesExtractor] = features_extractor.CmuHumanoidFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **ignored
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs['eps'] = 1e-5

        #features_extractor_kwargs = features_extractor_kwargs or dict()
        #features_extractor_kwargs['observable_keys'] = observables

        model.NpmpPolicy.__init__(self, observation_space, action_space, observables, ref_steps, np.nan, embed_size,
                         ref_encoder_n_layers, ref_encoder_layer_size, decoder_n_layers, decoder_layer_size, True,
                         layer_norm, ref_embedding_kl_weight, embedding_correlation,
                         bc_seq_steps, activation_fn, squash_output,
                         std_dev, features_extractor_class, features_extractor_kwargs, optimizer_class,
                         optimizer_kwargs)

        self.stand_vf_n_layers = stand_vf_n_layers
        self.stand_vf_layer_size = stand_vf_layer_size
        self.ortho_init = ortho_init
        self.log_std_init = log_std_init


        stand_vf_layers = torch_layers.create_mlp(
            self.features_extractor.sub_features_dim['decoder'],
            1,
            net_arch=self.stand_vf_n_layers*[self.stand_vf_layer_size],
            activation_fn=self.activation_fn,
            layer_norm=self.layer_norm
        )
        self.stand_vf = nn.Sequential(*stand_vf_layers)

        # TODO: clean up hack of having to overwrite the decoder
        action_decoder_layers = torch_layers.create_mlp(
            self.features_extractor.sub_features_dim['decoder'] + self.embed_size,
            self.action_space.shape[0],
            net_arch=self.decoder_n_layers*[self.decoder_layer_size],
            activation_fn=self.activation_fn,
            squash_output=self.squash_output,
            layer_norm=self.layer_norm
        )
        self.action_decoder = nn.Sequential(*action_decoder_layers)


        if self.ortho_init:
            self.action_decoder.apply(partial(self.init_weights, gain=np.sqrt(2)))

        self.log_std = nn.Parameter(torch.ones(self.action_space.shape[0])*self.log_std_init, requires_grad=True)
        #self.log_std = torch.ones(self.action_space.shape[0])*self.log_std_init
        #self.log_embed_std = nn.Parameter(torch.ones(self.embed_size)*self.log_embed_std_init, requires_grad=True)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = model.NpmpPolicy._get_constructor_parameters(self)
        data.update(
            dict(
                stand_vf_n_layers=self.stand_vf_n_layers,
                stand_vf_layer_size=self.stand_vf_layer_size,
                log_std_init=self.log_std_init,
                log_embed_std_init=self.log_embed_std_init
            )
        )
        return data

    def _predict(self, observation: torch.Tensor, state, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_features(observation)
        proprio, embed = features['decoder'], features['embed']
        act_distribution = self.action_decoder_forward(proprio, embed)
        return act_distribution.mean, state

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        proprio, embed = features['decoder'], features['embed']
        act_distribution = self.action_decoder_forward(proprio, embed)
        act_distribution = Independent(Normal(act_distribution.mean, self.log_std.exp()), 1)
        actions = act_distribution.mean if deterministic else act_distribution.sample()

        values = self.stand_vf(proprio)
        log_prob = act_distribution.log_prob(actions)

        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        proprio, embed = features['decoder'], features['embed']

        act_distribution = self.action_decoder_forward(proprio, embed)
        act_distribution = Independent(Normal(act_distribution.mean, self.log_std.exp()), 1)

        values = self.stand_vf(proprio)
        log_prob = act_distribution.log_prob(actions)
        return values, log_prob, act_distribution.entropy()

    def predict_values(self, obs):
        features = self.extract_features(obs)
        proprio = features['decoder']
        return self.stand_vf(proprio)

    def compute_bc_loss(self, obs, actions, weights):
        #features = self.extract_features(obs)
        features = obs
        references, proprios = features['ref_encoder'], features['decoder']
        B, T, _ = actions.shape
        embed = torch.randn(B, self.embed_size, dtype=actions.dtype, device=actions.device)

        total_kl, total_embed_std, total_delta_embed = 0, 0, 0
        all_embeds = [embed]
        for t in range(T):
            next_embed, next_embed_distribution = self.ref_encoder_forward(references[:, t], embed)
            prior_embed_distribution = Independent(Normal(self.embedding_correlation*embed, self.embedding_std_dev), 1)
            kl = kl_divergence(next_embed_distribution, prior_embed_distribution).sum()
            total_kl += kl
            with torch.no_grad():
                total_embed_std += next_embed_distribution.stddev.mean()
                total_delta_embed += torch.mean(torch.abs(next_embed_distribution.mean - self.embedding_correlation*embed))

            embed = next_embed
            all_embeds.append(embed)
        embeds = torch.stack(all_embeds, 1)

        act_distribution = self.action_decoder_forward(proprios, embeds[:, 1:])
        log_prob = act_distribution.log_prob(actions)
        weighted_log_prob = torch.einsum('ij,ij', weights, log_prob)
        loss = (-weighted_log_prob + self.embedding_kl_weight*total_kl) / (B*T)

        with torch.no_grad():
            mse = F.mse_loss(act_distribution.mean, actions)
            embed_mean = torch.mean(embeds)
            embed_std = torch.std(embeds)
        return loss, -weighted_log_prob.item() / (B*T), total_kl/(B*T), mse.item(), total_delta_embed.item()/T, total_embed_std.item()/T, embed_mean.item(), embed_std.item()
