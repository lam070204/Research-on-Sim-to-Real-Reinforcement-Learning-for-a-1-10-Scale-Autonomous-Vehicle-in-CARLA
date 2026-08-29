# -*- coding: utf-8 -*-
"""
GĐ14.2 - Canonical PPOAgent V3 with correct terminal/truncation handling.

Important:
- Physical terminal (collision/offroad/stuck):
      return target ends with 0 bootstrap.
- Time-limit truncation:
      return target bootstraps from V(next_obs).
- Mid-rollout PPO update:
      learn(last_obs=...) bootstraps from V(last_obs).

Action contract remains:
    steer in [-1,+1]
    speed in [0,1] m/s
"""

import os
import re
import numpy as np

import torch
import torch.nn as nn

from networks.on_policy.ppo.actor_critic_v3 import ActorCritic
from observation_builder_rgb_v3 import OBSERVATION_DIM_V3
from parameters_v3 import (
    ACTION_STD_INIT,
    GAMMA,
    POLICY_CLIP,
    PPO_CHECKPOINT_DIR,
    PPO_LEARNING_RATE,
)


DEFAULT_ACTION_DIM = 2
DEFAULT_K_EPOCHS = 7
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_VALUE_COEF = 0.5
DEFAULT_MAX_GRAD_NORM = 0.5


class Buffer:
    def __init__(self):
        self.observation = []
        self.raw_actions = []
        self.actions = self.raw_actions
        self.log_probs = []

        self.rewards = []
        self.terminateds = []
        self.truncateds = []
        self.bootstrap_values = []

        # Compatibility/debug convenience only.
        self.dones = []

    def clear(self):
        self.observation.clear()
        self.raw_actions.clear()
        self.log_probs.clear()

        self.rewards.clear()
        self.terminateds.clear()
        self.truncateds.clear()
        self.bootstrap_values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.rewards)


class PPOAgent(object):
    def __init__(
        self,
        town,
        action_std_init=ACTION_STD_INIT,
        device="cpu",
        k_epochs=DEFAULT_K_EPOCHS,
        entropy_coef=DEFAULT_ENTROPY_COEF,
        value_coef=DEFAULT_VALUE_COEF,
        max_grad_norm=DEFAULT_MAX_GRAD_NORM,
    ):
        self.obs_dim = OBSERVATION_DIM_V3
        self.action_dim = DEFAULT_ACTION_DIM

        self.clip = float(POLICY_CLIP)
        self.gamma = float(GAMMA)
        self.n_updates_per_iteration = int(k_epochs)
        self.lr = float(PPO_LEARNING_RATE)

        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)

        self.device = torch.device(device)
        self.action_std = float(action_std_init)

        self.memory = Buffer()
        self.town = str(town)
        self.checkpoint_file_no = 0

        self.policy = ActorCritic(
            self.obs_dim,
            self.action_dim,
            self.action_std,
        ).to(self.device)

        self.old_policy = ActorCritic(
            self.obs_dim,
            self.action_dim,
            self.action_std,
        ).to(self.device)

        self.old_policy.load_state_dict(
            self.policy.state_dict()
        )

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.lr,
        )

        self.value_loss_fn = nn.MSELoss()

    def _obs_tensor(self, obs):
        if isinstance(obs, torch.Tensor):
            tensor = obs.detach().to(
                device=self.device,
                dtype=torch.float32,
            )
        else:
            tensor = torch.as_tensor(
                np.asarray(obs),
                dtype=torch.float32,
                device=self.device,
            )

        tensor = tensor.reshape(-1)

        if tuple(tensor.shape) != (self.obs_dim,):
            raise ValueError(
                "PPO obs phải có shape ({},), nhận được {}.".format(
                    self.obs_dim,
                    tuple(tensor.shape),
                )
            )

        if not torch.isfinite(tensor).all():
            raise ValueError("PPO obs chứa NaN/Inf.")

        return tensor

    def _value_of_obs(self, obs):
        obs_tensor = self._obs_tensor(obs)

        with torch.no_grad():
            value = (
                self.old_policy
                .get_value(obs_tensor)
                .reshape(-1)[0]
            )

        value_float = float(
            value.detach().cpu()
        )

        if not np.isfinite(value_float):
            raise RuntimeError(
                "Bootstrap value NaN/Inf."
            )

        return value_float

    def get_action(self, obs, train=True):
        obs_tensor = self._obs_tensor(obs)

        if train:
            (
                env_action,
                raw_action,
                log_prob,
            ) = self.old_policy.act(
                obs_tensor
            )

            self.memory.observation.append(
                obs_tensor.detach().cpu()
            )
            self.memory.raw_actions.append(
                raw_action.detach().cpu()
            )
            self.memory.log_probs.append(
                log_prob.detach().cpu()
            )
        else:
            env_action = (
                self.old_policy
                .deterministic_action(
                    obs_tensor
                )
            )

        action_np = (
            env_action
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
            .reshape(2)
        )

        if not np.isfinite(action_np).all():
            raise RuntimeError(
                "PPO action chứa NaN/Inf: {}".format(
                    action_np
                )
            )

        if not (
            -1.0 <= float(action_np[0]) <= 1.0
        ):
            raise RuntimeError(
                "Steer action ngoài [-1,1]: {}".format(
                    action_np[0]
                )
            )

        if not (
            0.0 <= float(action_np[1]) <= 1.0
        ):
            raise RuntimeError(
                "Speed action ngoài [0,1]: {}".format(
                    action_np[1]
                )
            )

        return action_np

    def record_outcome(
        self,
        reward,
        done=None,
        terminated=None,
        truncated=False,
        next_obs=None,
    ):
        """
        Complete the transition for the most recently sampled train action.

        Preferred V3 call:
            agent.record_outcome(
                reward=reward,
                terminated=info["terminated"],
                truncated=info["truncated"],
                next_obs=next_obs,
            )

        Backward compatibility:
            record_outcome(reward, done=True)
        is interpreted as a true terminal unless truncated=True is supplied.
        """
        reward = float(reward)
        truncated = bool(truncated)

        if not np.isfinite(reward):
            raise ValueError(
                "Reward NaN/Inf: {}".format(reward)
            )

        if terminated is None:
            if done is None:
                terminated = False
            else:
                terminated = bool(done) and not truncated
        else:
            terminated = bool(terminated)

        if terminated and truncated:
            raise ValueError(
                "Một transition không thể vừa terminated vừa truncated."
            )

        expected = len(
            self.memory.observation
        )

        if len(self.memory.rewards) >= expected:
            raise RuntimeError(
                "record_outcome() được gọi nhiều hơn số action đã collect."
            )

        bootstrap_value = 0.0

        if truncated:
            if next_obs is None:
                raise ValueError(
                    "Time-limit truncation cần next_obs để bootstrap V(next_obs)."
                )

            bootstrap_value = self._value_of_obs(
                next_obs
            )

        self.memory.rewards.append(
            reward
        )
        self.memory.terminateds.append(
            terminated
        )
        self.memory.truncateds.append(
            truncated
        )
        self.memory.bootstrap_values.append(
            float(bootstrap_value)
        )
        self.memory.dones.append(
            bool(terminated or truncated)
        )

    def set_action_std(self, new_action_std):
        new_action_std = float(
            new_action_std
        )

        if new_action_std <= 0.0:
            raise ValueError(
                "new_action_std phải > 0."
            )

        self.action_std = new_action_std

        self.policy.set_action_std(
            new_action_std
        )
        self.old_policy.set_action_std(
            new_action_std
        )

    def decay_action_std(
        self,
        action_std_decay_rate,
        min_action_std,
    ):
        new_std = max(
            float(min_action_std),
            self.action_std
            - float(action_std_decay_rate),
        )

        self.set_action_std(
            new_std
        )

        return self.action_std

    def _validate_rollout(self):
        n = len(
            self.memory.observation
        )

        lengths = {
            "observations": n,
            "raw_actions": len(
                self.memory.raw_actions
            ),
            "log_probs": len(
                self.memory.log_probs
            ),
            "rewards": len(
                self.memory.rewards
            ),
            "terminateds": len(
                self.memory.terminateds
            ),
            "truncateds": len(
                self.memory.truncateds
            ),
            "bootstrap_values": len(
                self.memory.bootstrap_values
            ),
        }

        if n == 0:
            raise RuntimeError(
                "PPO learn(): rollout rỗng."
            )

        if len(set(lengths.values())) != 1:
            raise RuntimeError(
                "Rollout lengths không khớp: {}".format(
                    lengths
                )
            )

    def _discounted_returns(
        self,
        last_obs=None,
    ):
        """
        Correct returns across:
        - true terminals
        - time-limit truncations
        - optional mid-rollout cutoff
        """
        n = len(
            self.memory.rewards
        )

        if n <= 0:
            raise RuntimeError(
                "Không có reward để tính return."
            )

        final_is_boundary = bool(
            self.memory.terminateds[-1]
            or self.memory.truncateds[-1]
        )

        if final_is_boundary:
            discounted = 0.0
        else:
            if last_obs is None:
                raise ValueError(
                    "Rollout kết thúc giữa episode: learn(last_obs=...) "
                    "cần final next observation để bootstrap."
                )

            discounted = self._value_of_obs(
                last_obs
            )

        returns = []

        for (
            reward,
            terminated,
            truncated,
            bootstrap_value,
        ) in zip(
            reversed(self.memory.rewards),
            reversed(self.memory.terminateds),
            reversed(self.memory.truncateds),
            reversed(self.memory.bootstrap_values),
        ):
            reward = float(reward)

            if terminated:
                discounted = reward

            elif truncated:
                discounted = (
                    reward
                    + self.gamma
                    * float(bootstrap_value)
                )

            else:
                discounted = (
                    reward
                    + self.gamma
                    * discounted
                )

            returns.append(
                discounted
            )

        returns.reverse()

        return torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

    def learn(
        self,
        last_obs=None,
    ):
        self._validate_rollout()

        old_states = torch.stack(
            self.memory.observation,
            dim=0,
        ).to(
            device=self.device,
            dtype=torch.float32,
        )

        old_raw_actions = torch.stack(
            self.memory.raw_actions,
            dim=0,
        ).to(
            device=self.device,
            dtype=torch.float32,
        )

        old_log_probs = torch.stack(
            self.memory.log_probs,
            dim=0,
        ).to(
            device=self.device,
            dtype=torch.float32,
        ).reshape(-1)

        returns = self._discounted_returns(
            last_obs=last_obs
        )

        for name, tensor in (
            ("old_states", old_states),
            ("old_raw_actions", old_raw_actions),
            ("old_log_probs", old_log_probs),
            ("returns", returns),
        ):
            if not torch.isfinite(tensor).all():
                raise RuntimeError(
                    "{} chứa NaN/Inf.".format(
                        name
                    )
                )

        with torch.no_grad():
            old_values = (
                self.policy
                .critic(old_states)
                .squeeze(-1)
            )

            advantages = (
                returns - old_values
            )

            adv_mean = advantages.mean()
            adv_std = advantages.std(
                unbiased=False
            )

            advantages = (
                advantages - adv_mean
            ) / (
                adv_std + 1e-8
            )

        last_metrics = None

        for _ in range(
            self.n_updates_per_iteration
        ):
            (
                log_probs,
                values,
                entropy,
            ) = self.policy.evaluate(
                old_states,
                old_raw_actions,
            )

            ratios = torch.exp(
                log_probs
                - old_log_probs
            )

            surr1 = (
                ratios * advantages
            )

            surr2 = torch.clamp(
                ratios,
                1.0 - self.clip,
                1.0 + self.clip,
            ) * advantages

            policy_loss = -torch.min(
                surr1,
                surr2,
            ).mean()

            value_loss = (
                self.value_loss_fn(
                    values,
                    returns,
                )
            )

            entropy_mean = (
                entropy.mean()
            )

            total_loss = (
                policy_loss
                + self.value_coef
                * value_loss
                - self.entropy_coef
                * entropy_mean
            )

            if not torch.isfinite(
                total_loss
            ):
                raise RuntimeError(
                    "PPO loss NaN/Inf."
                )

            self.optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.max_grad_norm,
            )

            self.optimizer.step()

            last_metrics = {
                "loss": float(
                    total_loss.detach().cpu()
                ),
                "policy_loss": float(
                    policy_loss.detach().cpu()
                ),
                "value_loss": float(
                    value_loss.detach().cpu()
                ),
                "entropy": float(
                    entropy_mean.detach().cpu()
                ),
                "mean_return": float(
                    returns.mean().detach().cpu()
                ),
                "mean_advantage": float(
                    advantages.mean().detach().cpu()
                ),
            }

        self.old_policy.load_state_dict(
            self.policy.state_dict()
        )

        self.memory.clear()

        return last_metrics

    def _checkpoint_dir(self):
        directory = os.path.join(
            PPO_CHECKPOINT_DIR,
            self.town,
        )

        os.makedirs(
            directory,
            exist_ok=True,
        )

        return directory

    @staticmethod
    def _checkpoint_index(
        filename,
    ):
        match = re.match(
            r"ppo_policy_(\d+)_\.pth$",
            filename,
        )

        if match is None:
            return None

        return int(
            match.group(1)
        )

    def _checkpoint_files(self):
        directory = (
            self._checkpoint_dir()
        )
        items = []

        for name in os.listdir(
            directory
        ):
            index = self._checkpoint_index(
                name
            )

            if index is not None:
                items.append(
                    (
                        index,
                        os.path.join(
                            directory,
                            name,
                        ),
                    )
                )

        items.sort(
            key=lambda item: item[0]
        )

        return items

    def _save_to_path(
        self,
        path,
    ):
        torch.save(
            {
                "version": "PPO_V3_GD14",
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "action_std": self.action_std,
                "policy_state_dict":
                    self.old_policy.state_dict(),
                "optimizer_state_dict":
                    self.optimizer.state_dict(),
            },
            path,
        )

        return path

    def save(self):
        files = (
            self._checkpoint_files()
        )

        next_index = (
            files[-1][0] + 1
            if files
            else 0
        )

        path = os.path.join(
            self._checkpoint_dir(),
            "ppo_policy_{}_.pth".format(
                next_index
            ),
        )

        self.checkpoint_file_no = (
            next_index
        )

        return self._save_to_path(
            path
        )

    def chkpt_save(self):
        files = (
            self._checkpoint_files()
        )

        if files:
            index, path = files[-1]
        else:
            index = 0
            path = os.path.join(
                self._checkpoint_dir(),
                "ppo_policy_0_.pth",
            )

        self.checkpoint_file_no = (
            index
        )

        return self._save_to_path(
            path
        )

    def load(
        self,
        checkpoint_path=None,
    ):
        if checkpoint_path is None:
            files = (
                self._checkpoint_files()
            )

            if not files:
                raise FileNotFoundError(
                    "Không có PPO checkpoint trong {}.".format(
                        self._checkpoint_dir()
                    )
                )

            (
                index,
                checkpoint_path,
            ) = files[-1]

            self.checkpoint_file_no = (
                index
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        if not (
            isinstance(checkpoint, dict)
            and "policy_state_dict"
            in checkpoint
        ):
            raise RuntimeError(
                "Checkpoint cũ không tương thích PPO V3."
            )

        if checkpoint.get(
            "version"
        ) != "PPO_V3_GD14":
            raise RuntimeError(
                "Checkpoint không phải PPO V3 GĐ14."
            )

        if int(
            checkpoint.get(
                "obs_dim",
                -1,
            )
        ) != self.obs_dim:
            raise RuntimeError(
                "Checkpoint obs_dim không khớp."
            )

        if int(
            checkpoint.get(
                "action_dim",
                -1,
            )
        ) != self.action_dim:
            raise RuntimeError(
                "Checkpoint action_dim không khớp."
            )

        self.policy.load_state_dict(
            checkpoint[
                "policy_state_dict"
            ]
        )

        self.old_policy.load_state_dict(
            checkpoint[
                "policy_state_dict"
            ]
        )

        self.set_action_std(
            float(
                checkpoint.get(
                    "action_std",
                    self.action_std,
                )
            )
        )

        if (
            "optimizer_state_dict"
            in checkpoint
        ):
            self.optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        return checkpoint_path
