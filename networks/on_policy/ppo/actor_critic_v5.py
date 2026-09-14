# -*- coding: utf-8 -*-
"""
GĐ14 - Canonical PPO Actor-Critic V5.

Important action contract:
    raw_action ~ Normal(mean, std)

    env_action[0] = tanh(raw_action[0])          -> steer in [-1, +1]
    env_action[1] = (tanh(raw_action[1]) + 1)/2 -> speed in [0, 1] m/s

PPO stores/evaluates RAW Gaussian actions, while CARLA receives only the
transformed bounded actions. Therefore there is no action clipping mismatch
between sampled log-probability and the action distribution used for PPO.

This also removes the old stale-covariance bug:
- no cached cov_mat
- set_action_std() changes the actual std used by both sampling and evaluate()
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        action_std_init,
    ):
        super(ActorCritic, self).__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        if self.action_dim != 2:
            raise ValueError(
                "PPO V5 yêu cầu action_dim=2, nhận được {}.".format(
                    self.action_dim
                )
            )

        action_std_init = float(action_std_init)
        if action_std_init <= 0.0:
            raise ValueError("action_std_init phải > 0.")

        # Raw Gaussian mean. DO NOT put Tanh here: tanh is the explicit
        # environment-action transform below.
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Fixed exploration std, stored in state_dict and moved with .to().
        self.register_buffer(
            "action_std",
            torch.full(
                (self.action_dim,),
                action_std_init,
                dtype=torch.float32,
            ),
        )

    def forward(self):
        raise NotImplementedError

    def set_action_std(self, new_action_std):
        new_action_std = float(new_action_std)

        if new_action_std <= 0.0:
            raise ValueError("new_action_std phải > 0.")

        self.action_std.fill_(new_action_std)

    def _distribution(self, obs):
        mean = self.actor(obs)
        std = self.action_std.expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def transform_raw_action(raw_action):
        """
        Raw R^2 -> physical policy command domain.

        steer: [-1, +1]
        speed: [0, 1]
        """
        steer = torch.tanh(raw_action[..., 0])
        speed = 0.5 * (
            torch.tanh(raw_action[..., 1])
            + 1.0
        )

        return torch.stack(
            (steer, speed),
            dim=-1,
        )

    def get_value(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(
                obs,
                dtype=torch.float32,
            )

        return self.critic(obs)

    @torch.no_grad()
    def act(self, obs):
        """
        Stochastic training action.

        Returns:
            env_action : bounded [steer, speed]
            raw_action : unbounded Gaussian sample stored by PPO
            log_prob   : log p(raw_action)
        """
        dist = self._distribution(obs)

        raw_action = dist.sample()
        log_prob = dist.log_prob(
            raw_action
        ).sum(dim=-1)

        env_action = self.transform_raw_action(
            raw_action
        )

        return (
            env_action,
            raw_action,
            log_prob,
        )

    @torch.no_grad()
    def deterministic_action(self, obs):
        raw_mean = self.actor(obs)

        return self.transform_raw_action(
            raw_mean
        )

    def evaluate(self, obs, raw_action):
        """
        Evaluate the SAME raw Gaussian actions collected by old_policy.
        """
        dist = self._distribution(obs)

        log_probs = dist.log_prob(
            raw_action
        ).sum(dim=-1)

        entropy = dist.entropy().sum(
            dim=-1
        )

        values = self.critic(obs).squeeze(
            -1
        )

        return (
            log_probs,
            values,
            entropy,
        )

