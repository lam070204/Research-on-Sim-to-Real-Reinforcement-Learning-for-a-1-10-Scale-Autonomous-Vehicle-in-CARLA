import time
import random
import traceback
import numpy as np
import pygame
from simulation.connection import carla
from simulation.sensors import CameraSensor, CameraSensorEnv, CollisionSensor, RGBDatasetCamera

from simulation.settings import *


class CarlaEnvironment():

    def __init__(self, client, world, town, checkpoint_frequency=100, continuous_action=True) -> None:


        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.action_space = self.get_discrete_action_space()
        self.continous_action_space = continuous_action
        self.display_on = VISUAL_DISPLAY
        self.vehicle = None
        self.settings = None
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0
        self.fresh_start=True
        self.checkpoint_frequency = checkpoint_frequency
        self.route_waypoints = None
        self.town = town

        # Spawn manager: use every spawn point before reshuffling.
        self.spawn_points = list(self.map.get_spawn_points())
        if not self.spawn_points:
            raise RuntimeError(
                "Map hiện tại không có spawn point. "
                "Hãy kiểm tra OpenDRIVE/RoadRunner."
            )
        self.spawn_order = list(range(len(self.spawn_points)))
        random.shuffle(self.spawn_order)
        self.spawn_cursor = 0
        self.current_spawn_index = None

        # Give the vehicle time to settle on the road before PPO controls it.
        self.spawn_settle_seconds = 1.5
        self.spawn_grace_steps = 35
        
        # Objects to be kept alive
        self.camera_obj = None
        self.rgb_dataset_obj = None
        self.env_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None

        # Two very important lists for keeping track of our actors and their observations.
        self.sensor_list = list()
        self.actor_list = list()
        self.walker_list = list()
        self.create_pedestrians()



    # A reset function for reseting our environment.
    def reset(self):

        try:
            
            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                self.destroy_episode_actors()
            else:
                self.remove_sensors()

            # Blueprint of our main vehicle
            vehicle_bp = self.get_vehicle(CAR_NAME)

            # Spawn at a different point each episode. All spawn points are
            # used once before the order is shuffled again.
            self.vehicle, transform = self.spawn_vehicle_safely(vehicle_bp)
            self.actor_list.append(self.vehicle)

            # Let gravity place the vehicle on the road before sensors and PPO start.
            self.settle_spawned_vehicle()

            # Camera Sensor
            self.camera_obj = CameraSensor(self.vehicle)
            self.wait_for_camera_frame(self.camera_obj)
            self.image_obs = self.camera_obj.front_camera.pop(-1)
            self.sensor_list.append(self.camera_obj.sensor)

            # RGB camera used only to collect the new training dataset.
            # The semantic camera above remains the PPO observation source.
            self.rgb_dataset_obj = RGBDatasetCamera(
                self.vehicle,
                save_every=5,
                max_images=20000,
                test_interval=10,
            )
            self.sensor_list.append(self.rgb_dataset_obj.sensor)

            # Third person view of our vehicle in the Simulated env
            if self.display_on:
                self.env_camera_obj = CameraSensorEnv(self.vehicle)
                self.sensor_list.append(self.env_camera_obj.sensor)

            # Collision sensor
            self.collision_obj = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)

            
            self.timesteps = 0
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.previous_location = self.vehicle.get_location()
            self.distance_traveled = 0.0
            self.center_lane_deviation = 0.0
            self.target_speed = 20.0  # km/h
            self.max_speed = 30.0
            self.min_speed = 8.0
            self.max_distance_from_center = 3
            self.throttle = float(0.0)
            self.previous_steer = float(0.0)
            self.velocity = float(0.0)
            self.distance_from_center = float(0.0)
            self.angle = float(0.0)
            self.center_lane_deviation = 0.0
            self.distance_covered = 0.0


            # Always build a new route from the current spawn point.
            # This prevents a route from an old spawn being reused.
            self.current_waypoint_index = 0
            self.checkpoint_waypoint_index = 0
            self.fresh_start = True

            self.route_waypoints, route_distance = self.build_route_from_spawn(
                self.vehicle.get_location()
            )
            self.total_distance = max(len(self.route_waypoints) - 1, 1)

            print(
                "Spawn {}/{} | route: {} waypoints | {:.1f} m".format(
                    self.current_spawn_index + 1,
                    len(self.spawn_points),
                    len(self.route_waypoints),
                    route_distance,
                )
            )

            self.navigation_obs = np.array([self.throttle, self.velocity, self.previous_steer, self.distance_from_center, self.angle])

                        
            time.sleep(0.5)
            self.collision_history.clear()

            self.episode_start_time = time.time()
            return [self.image_obs, self.navigation_obs]

        except Exception as error:
            print("\n===== RESET ERROR =====")
            print(error)
            traceback.print_exc()
            print("=======================\n")
            self.destroy_episode_actors()
            if self.display_on:
                pygame.quit()
            raise


# ----------------------------------------------------------------
# Step method is used for implementing actions taken by our agent|
# ----------------------------------------------------------------

    # A step function is used for taking inputs generated by neural net.
    def step(self, action_idx):
        try:

            self.timesteps+=1
            self.fresh_start = True

            # Velocity of the vehicle
            velocity = self.vehicle.get_velocity()
            self.velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            
            # Action fron action space for contolling the vehicle with a discrete action
            if self.continous_action_space:
                steer = float(action_idx[0])
                steer = max(min(steer, 1.0), -1.0)
                throttle = float((action_idx[1] + 1.0) / 2)
                throttle = max(min(throttle, 1.0), 0.0)

                # Curve speed assist: reduce throttle when steering strongly.
                # This keeps the current checkpoint compatible while helping
                # the vehicle enter corners at a safer speed.
                abs_steer = abs(steer)
                if abs_steer > 0.45:
                    throttle *= 0.45
                elif abs_steer > 0.30:
                    throttle *= 0.65
                elif abs_steer > 0.15:
                    throttle *= 0.85

                smooth_steer = self.previous_steer * 0.9 + steer * 0.1
                smooth_throttle = self.throttle * 0.9 + throttle * 0.1
                brake = 0.0

                # Soft limiter: do not instantly terminate when the new policy
                # briefly commands too much throttle.
                if self.velocity >= self.max_speed:
                    smooth_throttle = 0.0
                    brake = 0.35
                elif self.velocity >= self.target_speed:
                    smooth_throttle = min(smooth_throttle, 0.15)

                # During the first frames after spawn, keep the car still.
                if self.timesteps <= self.spawn_grace_steps:
                    smooth_throttle = 0.0
                    brake = 1.0

                self.vehicle.apply_control(
                    carla.VehicleControl(
                        steer=smooth_steer,
                        throttle=smooth_throttle,
                        brake=brake,
                    )
                )
                self.previous_steer = steer
                self.throttle = throttle
            else:
                steer = self.action_space[action_idx]
                if self.velocity < 20.0:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, throttle=1.0))
                else:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1))
                self.previous_steer = steer
                self.throttle = 1.0
            
            # Traffic Light state
            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)

            self.collision_history = self.collision_obj.collision_data            

            # Rotation of the vehicle in correlation to the map/lane
            self.rotation = self.vehicle.get_transform().rotation.yaw

            # Location of the car
            self.location = self.vehicle.get_location()


            #transform = self.vehicle.get_transform()
            # Keep track of closest waypoint on the route
            waypoint_index = self.current_waypoint_index
            for _ in range(len(self.route_waypoints)):
                # Check if we passed the next waypoint along the route
                next_waypoint_index = waypoint_index + 1
                wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
                dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],self.vector(self.location - wp.transform.location)[:2])
                if dot > 0.0:
                    waypoint_index += 1
                else:
                    break

            self.current_waypoint_index = waypoint_index
            # Calculate deviation from center of the lane
            self.current_waypoint = self.route_waypoints[ self.current_waypoint_index    % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
            self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),self.vector(self.next_waypoint.transform.location),self.vector(self.location))
            self.center_lane_deviation += self.distance_from_center

            # Get angle difference between closest waypoint and vehicle forward vector
            fwd    = self.vector(self.vehicle.get_velocity())
            wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
            self.angle  = self.angle_diff(fwd, wp_fwd)

            # Route is rebuilt at every reset, so checkpoint teleporting is
            # intentionally disabled. Distance is measured from route start.
            self.checkpoint_waypoint_index = 0

            
            # Rewards are given below!
            done = False
            reward = 0
            done_reason = None

            # Never punish the vehicle while it is still settling after spawn.
            if self.timesteps <= self.spawn_grace_steps:
                self.collision_history.clear()
                reward = 0.0
            elif len(self.collision_history) != 0:
                done = True
                reward = -10
                done_reason = "COLLISION"
            elif self.distance_from_center > self.max_distance_from_center:
                done = True
                reward = -10
                done_reason = "OFF ROAD"
            elif self.episode_start_time + 10 < time.time() and self.velocity < 1.0:
                reward = -10
                done = True
                done_reason = "STUCK"
            elif self.velocity > self.max_speed + 3.0:
                # Keep only an emergency cutoff. Normal overspeed is handled
                # by the soft limiter above.
                reward = -5
                done = True
                done_reason = "EXTREME OVER SPEED"

            # Interpolated from 1 when centered to 0 when 3 m from center
            centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
            # Interpolated from 1 when aligned with the road to 0 when +/- 30 degress of road
            angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)

            if not done:
                if self.continous_action_space:
                    if self.velocity < self.min_speed:
                        speed_factor = self.velocity / self.min_speed
                    elif self.velocity > self.target_speed:
                        speed_factor = 1.0 - (
                            (self.velocity - self.target_speed)
                            / (self.max_speed - self.target_speed)
                        )
                        speed_factor = max(speed_factor, 0.0)
                    else:
                        speed_factor = 1.0

                    reward = 1.2 * speed_factor * centering_factor * angle_factor
                else:
                    reward = 1.2 * centering_factor * angle_factor

                # Heading bonus: reward the vehicle for pointing in the same
                # direction as the road, especially through curved sections.
                abs_angle = abs(self.angle)
                if abs_angle < np.deg2rad(5):
                    reward += 0.4
                elif abs_angle < np.deg2rad(10):
                    reward += 0.2

                # Early lane-deviation penalties. The episode still ends only
                # after crossing max_distance_from_center, but PPO receives an
                # earlier warning that the car is drifting toward the edge.
                if self.distance_from_center > 2.5:
                    reward -= 2.0
                elif self.distance_from_center > 2.0:
                    reward -= 1.0
                elif self.distance_from_center > 1.5:
                    reward -= 0.5

            if self.timesteps >= 50000:
                done = True
                done_reason = "TIME LIMIT"
            elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
                done = True
                self.fresh_start = True
                reward += 10.0
                done_reason = "ROUTE COMPLETED"

            self.wait_for_camera_frame(self.camera_obj)

            self.image_obs = self.camera_obj.front_camera.pop(-1)
            normalized_velocity = self.velocity/self.target_speed
            normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
            normalized_angle = abs(self.angle / np.deg2rad(20))
            self.navigation_obs = np.array([self.throttle, self.velocity, normalized_velocity, normalized_distance_from_center, normalized_angle])
            
            # Remove everything that has been spawned in the env
            if done:
                self.center_lane_deviation = (
                    self.center_lane_deviation / max(self.timesteps, 1)
                )
                self.distance_covered = abs(
                    self.current_waypoint_index -
                    self.checkpoint_waypoint_index
                )

                print(
                    "Episode ended: {} | speed: {:.2f} km/h | "
                    "lane distance: {:.2f} m | steps: {}".format(
                        done_reason or "UNKNOWN",
                        self.velocity,
                        self.distance_from_center,
                        self.timesteps,
                    )
                )

                self.destroy_episode_actors()

            return [self.image_obs, self.navigation_obs], reward, done, [self.distance_covered, self.center_lane_deviation]

        except Exception as error:
            print("\n===== STEP ERROR =====")
            print(error)
            traceback.print_exc()
            print("======================\n")
            self.destroy_episode_actors()
            if self.display_on:
                pygame.quit()
            raise



    # -------------------------------------------------
    # Spawn and automatic route helpers
    # -------------------------------------------------

    def settle_spawned_vehicle(self):
        """Allow gravity to place the newly spawned car on the road."""
        if self.vehicle is None:
            return

        try:
            self.vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    hand_brake=True,
                )
            )

            self.vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.vehicle.set_target_angular_velocity(
                carla.Vector3D(0.0, 0.0, 0.0)
            )

            settle_start = time.time()
            while time.time() - settle_start < self.spawn_settle_seconds:
                try:
                    self.world.wait_for_tick(1.0)
                except Exception:
                    time.sleep(0.05)

            self.vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    hand_brake=False,
                )
            )
        except Exception as error:
            print("WARNING settle_spawned_vehicle():", error)


    def get_next_spawn_transform(self):
        """Return the next spawn point without repeating points prematurely."""
        if self.spawn_cursor >= len(self.spawn_order):
            random.shuffle(self.spawn_order)
            self.spawn_cursor = 0

        index = self.spawn_order[self.spawn_cursor]
        self.spawn_cursor += 1
        self.current_spawn_index = index

        original = self.spawn_points[index]

        # Create a copy so the original map spawn point is not modified.
        return carla.Transform(
            carla.Location(
                x=original.location.x,
                y=original.location.y,
                z=original.location.z + 0.05,
            ),
            carla.Rotation(
                pitch=original.rotation.pitch,
                yaw=original.rotation.yaw,
                roll=original.rotation.roll,
            ),
        )

    def spawn_vehicle_safely(self, vehicle_bp):
        """Try each spawn point until the main vehicle is spawned."""
        for _ in range(len(self.spawn_points)):
            transform = self.get_next_spawn_transform()
            vehicle = self.world.try_spawn_actor(vehicle_bp, transform)

            if vehicle is not None:
                return vehicle, transform

            print("Spawn point {} is occupied/invalid".format(
                self.current_spawn_index + 1
            ))

        raise RuntimeError(
            "Không thể spawn xe tại bất kỳ spawn point nào của map."
        )

    def build_route_from_spawn(
        self,
        vehicle_location,
        step_distance=1.0,
        max_route_distance=1200.0,
        minimum_loop_distance=30.0,
        loop_close_distance=3.0,
    ):
        """
        Build a route from the current spawn.

        It supports simple open roads, closed loops and basic junctions.
        At junctions it usually keeps the straightest branch, while sometimes
        choosing another branch to increase training diversity.
        """
        start_waypoint = self.map.get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if start_waypoint is None:
            raise RuntimeError(
                "Không tìm thấy Driving waypoint tại spawn hiện tại."
            )

        route = [start_waypoint]
        current_waypoint = start_waypoint
        start_location = start_waypoint.transform.location
        previous_location = start_location
        traveled = 0.0
        visited = set()

        max_iterations = int(max_route_distance / max(step_distance, 0.1))

        for _ in range(max_iterations):
            key = (
                current_waypoint.road_id,
                current_waypoint.section_id,
                current_waypoint.lane_id,
                round(float(current_waypoint.s), 1),
            )

            if key in visited and traveled > minimum_loop_distance:
                break
            visited.add(key)

            candidates = current_waypoint.next(step_distance)
            if not candidates:
                break

            next_waypoint = self.choose_next_waypoint(
                current_waypoint, candidates
            )

            current_location = next_waypoint.transform.location
            traveled += current_location.distance(previous_location)

            route.append(next_waypoint)
            previous_location = current_location
            current_waypoint = next_waypoint

            if (
                traveled > minimum_loop_distance
                and current_location.distance(start_location)
                < loop_close_distance
            ):
                break

            if traveled >= max_route_distance:
                break

        if len(route) < 2:
            raise RuntimeError("Route tạo được quá ngắn.")

        return route, traveled

    def choose_next_waypoint(self, current_waypoint, candidates):
        """Choose a reasonable branch at a junction."""
        if len(candidates) == 1:
            return candidates[0]

        current_yaw = current_waypoint.transform.rotation.yaw
        scored = []

        for candidate in candidates:
            candidate_yaw = candidate.transform.rotation.yaw
            yaw_diff = abs(self.normalize_yaw(candidate_yaw - current_yaw))
            scored.append((yaw_diff, candidate))

        scored.sort(key=lambda item: item[0])

        # Mostly continue straight, but explore other valid branches too.
        if random.random() < 0.7:
            return scored[0][1]

        return random.choice(candidates)

    def normalize_yaw(self, yaw):
        while yaw > 180.0:
            yaw -= 360.0
        while yaw <= -180.0:
            yaw += 360.0
        return yaw


# -------------------------------------------------
# Creating and Spawning Pedestrians in our world |
# -------------------------------------------------

    # Walkers are to be included in the simulation yet!
    def create_pedestrians(self):
        try:

            # Our code for this method has been broken into 3 sections.

            # 1. Getting the available spawn points in  our world.
            # Random Spawn locations for the walker
            walker_spawn_points = []
            for i in range(NUMBER_OF_PEDESTRIAN):
                spawn_point_ = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point_.location = loc
                    walker_spawn_points.append(spawn_point_)

            # 2. We spawn the walker actor and ai controller
            # Also set their respective attributes
            for spawn_point_ in walker_spawn_points:
                walker_bp = random.choice(
                    self.blueprint_library.filter('walker.pedestrian.*'))
                walker_controller_bp = self.blueprint_library.find(
                    'controller.ai.walker')
                # Walkers are made visible in the simulation
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # They're all walking not running on their recommended speed
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute(
                        'speed', (walker_bp.get_attribute('speed').recommended_values[1]))
                else:
                    walker_bp.set_attribute('speed', 0.0)
                walker = self.world.try_spawn_actor(walker_bp, spawn_point_)
                if walker is not None:
                    walker_controller = self.world.spawn_actor(
                        walker_controller_bp, carla.Transform(), walker)
                    self.walker_list.append(walker_controller.id)
                    self.walker_list.append(walker.id)
            all_actors = self.world.get_actors(self.walker_list)

            # set how many pedestrians can cross the road
            #self.world.set_pedestrians_cross_factor(0.0)
            # 3. Starting the motion of our pedestrians
            for i in range(0, len(self.walker_list), 2):
                # start walker
                all_actors[i].start()
            # set walk to random point
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())

        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])


# ---------------------------------------------------
# Creating and Spawning other vehciles in our world|
# ---------------------------------------------------


    def set_other_vehicles(self):
        try:
            # NPC vehicles generated and set to autopilot
            # One simple for loop for creating x number of vehicles and spawing them into the world
            for _ in range(0, NUMBER_OF_VEHICLES):
                spawn_point = random.choice(self.map.get_spawn_points())
                bp_vehicle = random.choice(self.blueprint_library.filter('vehicle'))
                other_vehicle = self.world.try_spawn_actor(
                    bp_vehicle, spawn_point)
                if other_vehicle is not None:
                    other_vehicle.set_autopilot(True)
                    self.actor_list.append(other_vehicle)
            print("NPC vehicles have been generated in autopilot mode.")
        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list])


# ----------------------------------------------------------------
# Extra very important methods: their names explain their purpose|
# ----------------------------------------------------------------

    # Setter for changing the town on the server.
    def change_town(self, new_town):
        self.world = self.client.load_world(new_town)


    # Getter for fetching the current state of the world that simulator is in.
    def get_world(self) -> object:
        return self.world


    # Getter for fetching blueprint library of the simulator.
    def get_blueprint_library(self) -> object:
        return self.world.get_blueprint_library()


    # Action space of our vehicle. It can make eight unique actions.
    # Continuous actions are broken into discrete here!
    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi: angle -= 2 * np.pi
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle


    def distance_to_line(self, A, B, p):
        num   = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom


    def vector(self, v):
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])


    def get_discrete_action_space(self):
        action_space = \
            np.array([
            -0.50,
            -0.30,
            -0.10,
            0.0,
            0.10,
            0.30,
            0.50
            ])
        return action_space

    # Main vehicle blueprint method
    # It picks a random color for the vehicle everytime this method is called
    def get_vehicle(self, vehicle_name):
        blueprint = self.blueprint_library.filter(vehicle_name)[0]
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint


    # Spawn the vehicle in the environment
    def set_vehicle(self, vehicle_bp, spawn_points):
        # Main vehicle spawned into the env
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)


    def wait_for_camera_frame(self, camera_obj, timeout_seconds=10.0):
        """Chờ camera có frame nhưng không treo vô hạn."""
        start_time = time.time()
        while len(camera_obj.front_camera) == 0:
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(
                    "Không nhận được frame camera sau {:.1f} giây.".format(
                        timeout_seconds
                    )
                )
            time.sleep(0.0001)


    def destroy_episode_actors(self):
        """Dừng và xóa sensor/xe của episode hiện tại an toàn."""
        for sensor in list(self.sensor_list):
            try:
                if sensor is not None and sensor.is_alive:
                    sensor.stop()
            except Exception:
                pass

        for sensor in list(self.sensor_list):
            try:
                if sensor is not None and sensor.is_alive:
                    sensor.destroy()
            except Exception:
                pass
        self.sensor_list.clear()

        for actor in list(self.actor_list):
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        self.actor_list.clear()

        self.vehicle = None
        self.remove_sensors()

        try:
            self.world.wait_for_tick(1.0)
        except Exception:
            time.sleep(0.1)


    # Clean up method
    def remove_sensors(self):
        self.camera_obj = None
        self.rgb_dataset_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.env_camera_obj = None
        self.front_camera = None
        self.collision_history = None
        self.wrong_maneuver = None