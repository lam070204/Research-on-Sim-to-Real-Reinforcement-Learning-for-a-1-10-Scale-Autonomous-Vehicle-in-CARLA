# -*- coding: utf-8 -*-
from __future__ import print_function

from reward_manager_v3 import RewardManagerV3, EpisodeConfigV3
from reward_metrics_carla_v4 import RoadRewardMetricsV4

EpisodeConfigV4 = EpisodeConfigV3


class RewardManagerV4(RewardManagerV3):
    """
    Reuse RewardV3 + V3 termination.
    Only privileged road metric changes to wheel-aware V4.
    """

    def __init__(
        self,
        vehicle,
        reward_config=None,
        episode_config=None,
    ):
        super(RewardManagerV4, self).__init__(
            vehicle=vehicle,
            reward_config=reward_config,
            episode_config=episode_config,
        )

        # V3 road metric owns no CARLA actor, so replacement is safe.
        self.road_metrics = RoadRewardMetricsV4(vehicle)
        self.road_metrics.reset()

    def step(
        self,
        speed_mps,
        steer_cmd,
        speed_cmd_mps,
        dt,
        desired_speed_mps=None,
    ):
        reward, done, info = super(RewardManagerV4, self).step(
            speed_mps=speed_mps,
            steer_cmd=steer_cmd,
            speed_cmd_mps=speed_cmd_mps,
            dt=dt,
            desired_speed_mps=desired_speed_mps,
        )

        metric = self.road_metrics

        info["wheel_out_count"] = int(metric.wheel_out_count)
        info["wheel_out_flags"] = tuple(metric.wheel_out_flags)
        info["wheel_offroad_consecutive_ticks"] = int(
            metric.wheel_offroad_consecutive_ticks
        )
        info["wheel_offroad_confirm_ticks"] = int(
            metric.offroad_confirm_ticks
        )
        info["wheel_offroad_min_points_out"] = int(
            metric.offroad_min_points_out
        )
        info["wheel_point_source"] = str(metric.wheel_point_source)
        info["center_offroad_v3"] = bool(metric.center_offroad_v3)
        info["offroad_mode"] = "V4_2of4_points_3ticks"

        return reward, done, info
