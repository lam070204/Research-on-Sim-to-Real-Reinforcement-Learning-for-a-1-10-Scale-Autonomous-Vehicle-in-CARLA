import os
import sys
import time
import random
import numpy as np
import argparse
import logging
import pickle
import re
import torch
from distutils.util import strtobool
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from encoder_init_rgb_v2 import EncodeStateRGBV2
from networks.on_policy.ppo.agent_v2 import PPOAgent
from simulation.connection import ClientConnection
from simulation.environment_rgb_v2 import CarlaEnvironmentRGB
from simulation.sensors import (
    FRONT_CAMERA_WIDTH,
    FRONT_CAMERA_HEIGHT,
    FRONT_CAMERA_FPS,
    FRONT_CAMERA_FOV,
    FRONT_CAMERA_X,
    FRONT_CAMERA_Y,
    FRONT_CAMERA_Z,
    FRONT_CAMERA_PITCH,
    FRONT_CAMERA_YAW,
    FRONT_CAMERA_ROLL,
)
from parameters import *


OLD_SAFE_SPAWN_NUMBERS = [1, 4, 6, 7, 8, 9, 10, 11]


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, help='name of the experiment')
    parser.add_argument('--env-name', type=str, default='carla', help='name of the simulation environment')
    parser.add_argument('--learning-rate', type=float, default=PPO_LEARNING_RATE, help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=SEED, help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=TOTAL_TIMESTEPS, help='total timesteps of the experiment')
    parser.add_argument('--action-std-init', type=float, default=ACTION_STD_INIT, help='initial exploration noise')
    parser.add_argument('--test-timesteps', type=int, default=TEST_TIMESTEPS, help='timesteps to test our model')
    parser.add_argument('--episode-length', type=int, default=EPISODE_LENGTH, help='max timesteps in an episode')
    parser.add_argument('--train', default=True, type=boolean_string, help='is it training?')
    parser.add_argument('--town', type=str, default="Town07", help='which town do you like?')
    parser.add_argument('--load-checkpoint', type=boolean_string, default=MODEL_LOAD, help='resume training?')
    parser.add_argument(
        '--collect-rgb-dataset',
        type=boolean_string,
        default=False,
        help='collect RGB dataset while running?'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        default=os.path.join(
            'RGB_DATA_COLLECTION',
            'dataset_new_16000',
        ),
        help='output root for the new RGB dataset'
    )
    parser.add_argument(
        '--dataset-images-per-spawn',
        type=int,
        default=2000,
        help='number of RGB images saved for each old safe spawn'
    )
    parser.add_argument(
        '--dataset-save-every',
        type=int,
        default=10,
        help='save one image every N environment steps'
    )
    parser.add_argument(
        '--dataset-test-interval',
        type=int,
        default=10,
        help='every Nth saved image goes to the test split'
    )
    parser.add_argument(
        '--dataset-min-speed',
        type=float,
        default=1.0,
        help='do not save images below this vehicle speed in km/h'
    )
    parser.add_argument(
        '--dataset-max-environment-steps',
        type=int,
        default=500000,
        help='safety limit for data-collection environment steps'
    )
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, cuda will not be enabled by deafult')
    args = parser.parse_args()
    
    return args

def boolean_string(value):
    """Parse common command-line boolean values safely."""
    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()
    if value in {'true', '1', 'yes', 'y', 'on'}:
        return True
    if value in {'false', '0', 'no', 'n', 'off'}:
        return False

    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value}. Use true or false."
    )



class RGBDatasetPerSpawn:
    """Save exactly 2,000 RGB images for every old safe spawn."""

    FILE_PATTERN = re.compile(
        r"^rgb_(\d+)_spawn_(\d+)_episode_(\d+)_step_(\d+)\.png$"
    )

    def __init__(
        self,
        root,
        safe_spawn_numbers,
        images_per_spawn,
        save_every,
        test_interval,
        min_speed_kmh,
    ):
        self.root = os.path.abspath(root)
        self.train_dir = os.path.join(
            self.root,
            "train",
            "rgb",
        )
        self.test_dir = os.path.join(
            self.root,
            "test",
            "rgb",
        )

        self.safe_spawn_numbers = [
            int(number) for number in safe_spawn_numbers
        ]
        self.images_per_spawn = max(
            int(images_per_spawn),
            1,
        )
        self.save_every = max(int(save_every), 1)
        self.test_interval = max(
            int(test_interval),
            2,
        )
        self.min_speed_kmh = max(
            float(min_speed_kmh),
            0.0,
        )

        self.spawn_counts = {
            number: 0
            for number in self.safe_spawn_numbers
        }
        self.next_index = 0

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

        self._scan_existing_images()
        self._write_manifest()

    @property
    def target_total(self):
        return (
            len(self.safe_spawn_numbers)
            * self.images_per_spawn
        )

    @property
    def saved_total(self):
        return sum(self.spawn_counts.values())

    @property
    def complete(self):
        return self.saved_total >= self.target_total

    def spawn_count(self, spawn_number):
        return self.spawn_counts.get(
            int(spawn_number),
            0,
        )

    def spawn_complete(self, spawn_number):
        return (
            self.spawn_count(spawn_number)
            >= self.images_per_spawn
        )

    def _iter_png_files(self):
        for directory in (
            self.train_dir,
            self.test_dir,
        ):
            if not os.path.isdir(directory):
                continue

            for name in os.listdir(directory):
                if name.lower().endswith(".png"):
                    yield name

    def _scan_existing_images(self):
        max_index = -1
        invalid_files = []

        for name in self._iter_png_files():
            match = self.FILE_PATTERN.match(name)

            if match is None:
                invalid_files.append(name)
                continue

            index = int(match.group(1))
            spawn_number = int(match.group(2))

            if spawn_number not in self.spawn_counts:
                invalid_files.append(name)
                continue

            self.spawn_counts[spawn_number] += 1
            max_index = max(max_index, index)

        if invalid_files:
            raise RuntimeError(
                "The NEW dataset contains files from another collector. "
                "Run the one-time reset script before collecting. "
                "Examples: {}".format(invalid_files[:5])
            )

        for spawn_number, count in self.spawn_counts.items():
            if count > self.images_per_spawn:
                raise RuntimeError(
                    "Spawn {} already contains {} images, "
                    "which is greater than the target {}."
                    .format(
                        spawn_number,
                        count,
                        self.images_per_spawn,
                    )
                )

        self.next_index = max_index + 1

    def _write_manifest(self):
        path = os.path.join(
            self.root,
            "DATASET_INFO.txt",
        )

        lines = [
            "RGB DATASET FROM continuous_driver_rgb_v2.py",
            "",
            "Camera source: CarlaEnvironmentRGB / CameraSensorRGBPPO",
            "Camera type: sensor.camera.rgb",
            "Width: {}".format(FRONT_CAMERA_WIDTH),
            "Height: {}".format(FRONT_CAMERA_HEIGHT),
            "FPS: {}".format(FRONT_CAMERA_FPS),
            "FOV: {}".format(FRONT_CAMERA_FOV),
            "X: {}".format(FRONT_CAMERA_X),
            "Y: {}".format(FRONT_CAMERA_Y),
            "Z: {}".format(FRONT_CAMERA_Z),
            "Pitch: {}".format(FRONT_CAMERA_PITCH),
            "Yaw: {}".format(FRONT_CAMERA_YAW),
            "Roll: {}".format(FRONT_CAMERA_ROLL),
            "",
            "Safe spawns: {}".format(
                self.safe_spawn_numbers
            ),
            "Images per spawn: {}".format(
                self.images_per_spawn
            ),
            "Target total: {}".format(
                self.target_total
            ),
            "Save every environment steps: {}".format(
                self.save_every
            ),
            "Test interval: {}".format(
                self.test_interval
            ),
            "Minimum speed km/h: {}".format(
                self.min_speed_kmh
            ),
            "PPO model namespace: mapden_rgb_v2",
            "PPO learning while collecting: False",
            "Weather: current CARLA world weather",
        ]

        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))

    def save(
        self,
        rgb_image,
        spawn_number,
        episode,
        environment_step,
        speed_kmh,
    ):
        spawn_number = int(spawn_number)

        if self.complete:
            return False

        if spawn_number not in self.spawn_counts:
            return False

        if self.spawn_complete(spawn_number):
            return False

        if environment_step % self.save_every != 0:
            return False

        if float(speed_kmh) < self.min_speed_kmh:
            return False

        image_array = np.asarray(rgb_image)

        if (
            image_array.ndim != 3
            or image_array.shape[2] != 3
        ):
            raise ValueError(
                "Expected an RGB image with shape (H,W,3), "
                "received {}.".format(image_array.shape)
            )

        if image_array.dtype != np.uint8:
            image_array = np.nan_to_num(
                image_array,
                nan=0.0,
                posinf=255.0,
                neginf=0.0,
            )

            if image_array.max() <= 1.0:
                image_array = image_array * 255.0

            image_array = np.clip(
                image_array,
                0.0,
                255.0,
            ).astype(np.uint8)

        index = self.next_index

        filename = (
            "rgb_{:06d}_spawn_{:02d}_"
            "episode_{:05d}_step_{:08d}.png"
        ).format(
            index,
            spawn_number,
            int(episode),
            int(environment_step),
        )

        output_dir = (
            self.test_dir
            if index % self.test_interval == 0
            else self.train_dir
        )

        Image.fromarray(
            image_array,
            mode="RGB",
        ).save(
            os.path.join(output_dir, filename)
        )

        self.next_index += 1
        self.spawn_counts[spawn_number] += 1

        if (
            self.saved_total % 100 == 0
            or self.complete
            or self.spawn_complete(spawn_number)
        ):
            print(
                "\n[RGB DATA] total {}/{} | "
                "spawn {}: {}/{} | speed {:.1f} km/h"
                .format(
                    self.saved_total,
                    self.target_total,
                    spawn_number,
                    self.spawn_count(spawn_number),
                    self.images_per_spawn,
                    float(speed_kmh),
                )
            )

        return True

    def print_status(self):
        print("\n" + "=" * 64)
        print("RGB DATASET - 2,000 IMAGES PER OLD SAFE SPAWN")
        print("=" * 64)

        for spawn_number in self.safe_spawn_numbers:
            print(
                "Spawn {:02d}: {:4d}/{}".format(
                    spawn_number,
                    self.spawn_count(spawn_number),
                    self.images_per_spawn,
                )
            )

        print("-" * 64)
        print(
            "TOTAL: {}/{}".format(
                self.saved_total,
                self.target_total,
            )
        )
        print("DATASET:", self.root)
        print("=" * 64)


def collect_rgb_dataset_with_trained_ppo(
    args,
    env,
    encode,
    agent,
):
    dataset = RGBDatasetPerSpawn(
        root=args.dataset_root,
        safe_spawn_numbers=OLD_SAFE_SPAWN_NUMBERS,
        images_per_spawn=args.dataset_images_per_spawn,
        save_every=args.dataset_save_every,
        test_interval=args.dataset_test_interval,
        min_speed_kmh=args.dataset_min_speed,
    )

    print("\n===== RGB COLLECTION MODE =====")
    print("Driver       : continuous_driver_rgb_v2.py")
    print("PPO          : mapden_rgb_v2, inference only")
    print("Safe spawns  :", OLD_SAFE_SPAWN_NUMBERS)
    print("Images/spawn :", dataset.images_per_spawn)
    print("Target total :", dataset.target_total)
    print(
        "Camera       : {}x{} @ {} FPS | FOV {} | "
        "XYZ=({:.6f}, {:.6f}, {:.6f}) | "
        "rotation=({:.1f}, {:.1f}, {:.1f})"
        .format(
            FRONT_CAMERA_WIDTH,
            FRONT_CAMERA_HEIGHT,
            FRONT_CAMERA_FPS,
            FRONT_CAMERA_FOV,
            FRONT_CAMERA_X,
            FRONT_CAMERA_Y,
            FRONT_CAMERA_Z,
            FRONT_CAMERA_PITCH,
            FRONT_CAMERA_YAW,
            FRONT_CAMERA_ROLL,
        )
    )
    print("Weather      : current CARLA world weather")
    print("PPO training : disabled")
    print("================================\n")
    dataset.print_status()

    if dataset.complete:
        print("The NEW dataset is already complete.")
        return

    episode = 0
    environment_step = 0

    try:
        while (
            not dataset.complete
            and environment_step
            < args.dataset_max_environment_steps
        ):
            raw_observation = env.reset()

            spawn_number = (
                int(
                    getattr(
                        env,
                        "current_spawn_index",
                        -1,
                    )
                )
                + 1
            )

            print(
                "\nEpisode {} | spawn {} | progress {}/{} | "
                "total {}/{}".format(
                    episode,
                    spawn_number,
                    dataset.spawn_count(spawn_number),
                    dataset.images_per_spawn,
                    dataset.saved_total,
                    dataset.target_total,
                )
            )

            if dataset.spawn_complete(spawn_number):
                env.destroy_episode_actors()
                episode += 1
                continue

            observation = encode.process(raw_observation)
            actors_already_destroyed = False

            for _ in range(args.episode_length):
                action = agent.get_action(
                    observation,
                    train=False,
                )

                (
                    raw_observation,
                    reward,
                    done,
                    info,
                ) = env.step(action)

                if raw_observation is None:
                    break

                environment_step += 1

                dataset.save(
                    rgb_image=raw_observation[0],
                    spawn_number=spawn_number,
                    episode=episode,
                    environment_step=environment_step,
                    speed_kmh=float(
                        getattr(env, "velocity", 0.0)
                    ),
                )

                if done:
                    actors_already_destroyed = True

                if (
                    dataset.complete
                    or dataset.spawn_complete(
                        spawn_number
                    )
                ):
                    break

                observation = encode.process(
                    raw_observation
                )

                if done:
                    break

            if not actors_already_destroyed:
                env.destroy_episode_actors()

            episode += 1

        dataset.print_status()

        if dataset.complete:
            print("\nCOLLECTION COMPLETE: 16,000 RGB images.")
        else:
            print(
                "\nCollection stopped at the environment-step "
                "safety limit: {}.".format(
                    args.dataset_max_environment_steps
                )
            )

    except KeyboardInterrupt:
        print("\nStopped by user. Existing images were kept.")
        dataset.print_status()

    finally:
        try:
            env.destroy_episode_actors()
        except Exception:
            pass



def runner():

    #========================================================================
    #                           BASIC PARAMETER & LOGGING SETUP
    #========================================================================
    
    args = parse_args()
    exp_name = args.exp_name
    train = args.train
    if args.collect_rgb_dataset:
        train = False
    town = args.town
    model_town = town + '_rgb_v2'
    checkpoint_load = args.load_checkpoint
    total_timesteps = args.total_timesteps
    action_std_init = args.action_std_init

    checkpoint_dir = os.path.join("checkpoints", "PPO", model_town)
    pretrained_dir = os.path.join("preTrained_models", "PPO", model_town)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(pretrained_dir, exist_ok=True)

    print("Checkpoint directory:", checkpoint_dir)
    print("Pretrained model directory:", pretrained_dir)

    print(
        f"Mode: {'TRAIN' if train else 'TEST / DATA COLLECTION'} | "
        f"Town: {town} | Load checkpoint: {checkpoint_load} | "
        f"Episode length: {args.episode_length} | "
        f"Test timesteps: {args.test_timesteps} | "
        f"Collect RGB: {args.collect_rgb_dataset}"
    )

    try:
        if exp_name == 'ppo':
            run_name = "PPO"
        else:
            """
            
            Here the functionality can be extended to different algorithms.

            """ 
            sys.exit() 
    except Exception as e:
        print(e.message)
        sys.exit()
    
    writer = None
    if not args.collect_rgb_dataset:
        if train == True:
            writer = SummaryWriter(
                f"runs/{run_name}_{action_std_init}_{int(total_timesteps)}/{town}"
            )
        else:
            writer = SummaryWriter(
                f"runs/{run_name}_{action_std_init}_{int(total_timesteps)}_TEST/{town}"
            )
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join(
                    [
                        f"|{key}|{value}"
                        for key, value in vars(args).items()
                    ]
                )
            ),
        )


    #Seeding to reproduce the results 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    
    action_std_decay_rate = 0.05
    min_action_std = 0.05   
    action_std_decay_freq = 5e5
    timestep = 0
    episode = 0
    cumulative_score = 0
    episodic_length = list()
    scores = list()
    deviation_from_center = 0
    distance_covered = 0

    #========================================================================
    #                           CREATING THE SIMULATION
    #========================================================================

    try:
        client, world = ClientConnection(town).setup()
        logging.info("Connection has been setup successfully.")
    except Exception:
        logging.exception("Connection has been refused by the server.")
        raise
    env = CarlaEnvironmentRGB(
        client,
        world,
        town,
        checkpoint_frequency=100 if train else None,
        collect_rgb_dataset=False,
        safe_spawn_numbers=OLD_SAFE_SPAWN_NUMBERS,
    )
    encode = EncodeStateRGBV2(LATENT_DIM)


    #========================================================================
    #                           ALGORITHM
    #========================================================================
    try:
        time.sleep(0.5)
        
        if checkpoint_load:
            checkpoint_dir = f'checkpoints/PPO/{model_town}'
            checkpoint_files = [
                name for name in os.listdir(checkpoint_dir)
                if name.startswith('checkpoint_ppo_') and name.endswith('.pickle')
            ]

            if not checkpoint_files:
                raise FileNotFoundError(
                    f'No PPO checkpoint metadata found in: {checkpoint_dir}'
                )

            checkpoint_files.sort(
                key=lambda name: int(
                    name.replace('checkpoint_ppo_', '').replace('.pickle', '')
                )
            )
            chkpt_file = os.path.join(checkpoint_dir, checkpoint_files[-1])

            with open(chkpt_file, 'rb') as f:
                data = pickle.load(f)

            action_std_init = data.get('action_std_init', action_std_init)
            agent = PPOAgent(model_town, action_std_init)
            agent.load()

            if train:
                episode = data.get('episode', 0)
                timestep = data.get('timestep', 0)
                cumulative_score = data.get('cumulative_score', 0)
            else:
                # Testing/data collection starts from zero instead of inheriting
                # the large training timestep stored in the checkpoint.
                episode = 0
                timestep = 0
                cumulative_score = 0

                for params in agent.old_policy.actor.parameters():
                    params.requires_grad = False
        else:
            if train == False:
                agent = PPOAgent(model_town, action_std_init)
                agent.load()
                for params in agent.old_policy.actor.parameters():
                    params.requires_grad = False
            else:
                agent = PPOAgent(model_town, action_std_init)

        if args.collect_rgb_dataset:
            if not checkpoint_load:
                raise RuntimeError(
                    "RGB collection requires --load-checkpoint true."
                )

            collect_rgb_dataset_with_trained_ppo(
                args=args,
                env=env,
                encode=encode,
                agent=agent,
            )
            return

        if train:
            #Training
            while timestep < total_timesteps:
            
                observation = env.reset()
                observation = encode.process(observation)

                current_ep_reward = 0
                episode_speeds = []
                episode_lane_distances = []
                episode_steers = []
                episode_throttles = []
                episode_rewards = []
                episode_done_reason = "EPISODE LENGTH"
                t1 = datetime.now()

                for t in range(args.episode_length):
                
                    # select action with policy
                    action = agent.get_action(observation, train=True)

                    observation, reward, done, info = env.step(action)
                    if observation is None:
                        break
                    observation = encode.process(observation)
                    
                    agent.memory.rewards.append(reward)
                    agent.memory.dones.append(done)
                    
                    timestep += 1
                    current_ep_reward += reward

                    episode_rewards.append(float(reward))
                    episode_speeds.append(float(getattr(env, "velocity", 0.0)))
                    episode_lane_distances.append(
                        float(getattr(env, "distance_from_center", 0.0))
                    )
                    episode_steers.append(
                        float(getattr(env, "previous_steer", 0.0))
                    )
                    episode_throttles.append(
                        float(getattr(env, "throttle", 0.0))
                    )

                    if done:
                        collision_history = getattr(env, "collision_history", None)
                        lane_now = float(getattr(env, "distance_from_center", 0.0))
                        speed_now = float(getattr(env, "velocity", 0.0))
                        max_lane_allowed = float(
                            getattr(env, "max_distance_from_center", 3.0)
                        )
                        max_speed_allowed = float(
                            getattr(env, "max_speed", 30.0)
                        )

                        if collision_history:
                            episode_done_reason = "COLLISION"
                        elif lane_now > max_lane_allowed:
                            episode_done_reason = "OFF ROAD"
                        elif speed_now > max_speed_allowed + 3.0:
                            episode_done_reason = "EXTREME OVER SPEED"
                        else:
                            episode_done_reason = "DONE"
                    
                    if timestep % action_std_decay_freq == 0:
                        action_std_init =  agent.decay_action_std(action_std_decay_rate, min_action_std)

                    if timestep == total_timesteps -1:
                        agent.chkpt_save()

                    # break; if the episode is over
                    if done:
                        episode += 1

                        t2 = datetime.now()
                        t3 = t2-t1
                        
                        episodic_length.append(abs(t3.total_seconds()))
                        break
                
                deviation_from_center += info[1]
                distance_covered += info[0]
                
                scores.append(current_ep_reward)
                
                if checkpoint_load:
                    cumulative_score = ((cumulative_score * (episode - 1)) + current_ep_reward) / (episode)
                else:
                    cumulative_score = np.mean(scores)


                ep_steps = len(episode_rewards)
                avg_speed = float(np.mean(episode_speeds)) if episode_speeds else 0.0
                max_speed_ep = float(np.max(episode_speeds)) if episode_speeds else 0.0
                avg_lane = float(np.mean(episode_lane_distances)) if episode_lane_distances else 0.0
                max_lane = float(np.max(episode_lane_distances)) if episode_lane_distances else 0.0
                avg_abs_steer = float(np.mean(np.abs(episode_steers))) if episode_steers else 0.0
                avg_throttle = float(np.mean(episode_throttles)) if episode_throttles else 0.0
                avg_step_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0

                print(
                    "\n===== EPISODE MONITOR =====\n"
                    "Episode          : {}\n"
                    "Total timestep    : {} / {}\n"
                    "Episode steps     : {}\n"
                    "End reason        : {}\n"
                    "Episode reward    : {:.2f}\n"
                    "Average reward    : {:.2f}\n"
                    "Reward / step     : {:.4f}\n"
                    "Average speed     : {:.2f} km/h\n"
                    "Maximum speed     : {:.2f} km/h\n"
                    "Average lane dist : {:.3f} m\n"
                    "Maximum lane dist : {:.3f} m\n"
                    "Average |steer|   : {:.4f}\n"
                    "Average throttle  : {:.4f}\n"
                    "Spawn             : {}/{}\n"
                    "===========================\n".format(
                        episode,
                        timestep,
                        total_timesteps,
                        ep_steps,
                        episode_done_reason,
                        current_ep_reward,
                        cumulative_score,
                        avg_step_reward,
                        avg_speed,
                        max_speed_ep,
                        avg_lane,
                        max_lane,
                        avg_abs_steer,
                        avg_throttle,
                        int(getattr(env, "current_spawn_index", -1)) + 1,
                        len(getattr(env, "spawn_points", [])),
                    )
                )
                if episode % 10 == 0:
                    agent.learn()
                    agent.chkpt_save()
                    chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{model_town}'))[2])
                    if chkt_file_nums != 0:
                        chkt_file_nums -=1
                    chkpt_file = f'checkpoints/PPO/{model_town}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
                    data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep, 'action_std_init': action_std_init}
                    with open(chkpt_file, 'wb') as handle:
                        pickle.dump(data_obj, handle)
                    
                
                if episode % 5 == 0:

                    writer.add_scalar("Episodic Reward/episode", scores[-1], episode)
                    writer.add_scalar("Cumulative Reward/info", cumulative_score, episode)
                    writer.add_scalar("Cumulative Reward/(t)", cumulative_score, timestep)
                    writer.add_scalar("Average Episodic Reward/info", np.mean(scores[-5]), episode)
                    writer.add_scalar("Average Reward/(t)", np.mean(scores[-5]), timestep)
                    writer.add_scalar("Episode Length (s)/info", np.mean(episodic_length), episode)
                    writer.add_scalar("Reward/(t)", current_ep_reward, timestep)
                    writer.add_scalar("Average Deviation from Center/episode", deviation_from_center/5, episode)
                    writer.add_scalar("Average Deviation from Center/(t)", deviation_from_center/5, timestep)
                    writer.add_scalar("Average Distance Covered (m)/episode", distance_covered/5, episode)
                    writer.add_scalar("Average Distance Covered (m)/(t)", distance_covered/5, timestep)

                    writer.add_scalar("Monitor/Average Speed kmh", avg_speed, episode)
                    writer.add_scalar("Monitor/Maximum Speed kmh", max_speed_ep, episode)
                    writer.add_scalar("Monitor/Average Lane Distance m", avg_lane, episode)
                    writer.add_scalar("Monitor/Maximum Lane Distance m", max_lane, episode)
                    writer.add_scalar("Monitor/Average Abs Steer", avg_abs_steer, episode)
                    writer.add_scalar("Monitor/Average Throttle", avg_throttle, episode)
                    writer.add_scalar("Monitor/Reward Per Step", avg_step_reward, episode)
                    writer.add_scalar("Monitor/Episode Steps", ep_steps, episode)
                    writer.add_scalar(
                        "Monitor/Spawn Number",
                        int(getattr(env, "current_spawn_index", -1)) + 1,
                        episode,
                    )

                    episodic_length = list()
                    deviation_from_center = 0
                    distance_covered = 0

                if episode % 100 == 0:
                    
                    agent.save()
                    chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{model_town}'))[2])
                    chkpt_file = f'checkpoints/PPO/{model_town}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
                    data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep, 'action_std_init': action_std_init}
                    with open(chkpt_file, 'wb') as handle:
                        pickle.dump(data_obj, handle)
                        
            print("Terminating the run.")
            sys.exit()
        else:
            #Testing
            while timestep < args.test_timesteps:
                observation = env.reset()
                observation = encode.process(observation)

                current_ep_reward = 0
                t1 = datetime.now()
                for t in range(args.episode_length):
                    # select action with policy
                    action = agent.get_action(observation, train=False)
                    observation, reward, done, info = env.step(action)
                    if observation is None:
                        break
                    observation = encode.process(observation)
                    
                    timestep +=1
                    current_ep_reward += reward
                    # break; if the episode is over
                    if done:
                        episode += 1

                        t2 = datetime.now()
                        t3 = t2-t1
                        
                        episodic_length.append(abs(t3.total_seconds()))
                        break
                deviation_from_center += info[1]
                distance_covered += info[0]
                
                scores.append(current_ep_reward)
                cumulative_score = np.mean(scores)

                print('Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score))
                
                writer.add_scalar("TEST: Episodic Reward/episode", scores[-1], episode)
                writer.add_scalar("TEST: Cumulative Reward/info", cumulative_score, episode)
                writer.add_scalar("TEST: Cumulative Reward/(t)", cumulative_score, timestep)
                writer.add_scalar("TEST: Episode Length (s)/info", np.mean(episodic_length), episode)
                writer.add_scalar("TEST: Reward/(t)", current_ep_reward, timestep)
                writer.add_scalar("TEST: Deviation from Center/episode", deviation_from_center, episode)
                writer.add_scalar("TEST: Deviation from Center/(t)", deviation_from_center, timestep)
                writer.add_scalar("TEST: Distance Covered (m)/episode", distance_covered, episode)
                writer.add_scalar("TEST: Distance Covered (m)/(t)", distance_covered, timestep)

                episodic_length = list()
                deviation_from_center = 0
                distance_covered = 0

            print("Terminating the run.")
            sys.exit()

    except Exception:
        import traceback
        print("\n===== TRAINING REAL ERROR =====")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        runner()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception:
        # The full traceback has already been printed above when the error
        # happens inside the training/testing loop.
        pass
    finally:
        print("\nExit")