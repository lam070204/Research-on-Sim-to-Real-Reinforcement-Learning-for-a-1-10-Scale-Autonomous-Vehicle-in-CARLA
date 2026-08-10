# 07 Environment

## CARLA Environment

Tài liệu về Environment class quản lý tương tác giữa agent và CARLA simulator.

## Tổng quan

Class `CarlaEnvironmentRGB` trong `simulation/environment_rgb_v2.py` là environment chính cho training PPO với RGB camera.

## Class CarlaEnvironmentRGB

### Khởi tạo

```python
class CarlaEnvironmentRGB:
    def __init__(self, town, autopilot=False):
        self.town = town
        self.autopilot = autopilot
        self.client = None
        self.world = None
        self.map = None
        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.spawned_actors = []
        self.safe_spawn_points = [1, 4, 6, 7, 8, 9, 10, 11]
        self.current_spawn_index = 0
        self.stuck_counter = 0
        self.previous_location = None
        self.route = []
        self.current_waypoint_index = 0
```

### Các phương thức chính

#### 1. `create_carla_connection()`
Thiết lập kết nối với CARLA server.

```python
def create_carla_connection(self):
    self.client = carla.Client("localhost", 2000)
    self.client.set_timeout(10.0)
    self.world = self.client.get_world()
    self.map = self.world.get_map()
```

#### 2. `spawn_vehicle()`
Spawn vehicle tại spawn point an toàn.

```python
def spawn_vehicle(self):
    if self.current_spawn_index >= len(self.safe_spawn_points):
        self.current_spawn_index = 0
        random.shuffle(self.safe_spawn_points)
    
    spawn_point_index = self.safe_spawn_points[self.current_spawn_index]
    spawn_points = self.map.get_spawn_points()
    spawn_point = spawn_points[spawn_point_index]
    
    blueprint_library = self.world.get_blueprint_library()
    vehicle_bp = random.choice(blueprint_library.filter("vehicle.lincoln*"))
    self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
    self.spawned_actors.append(self.vehicle)
    
    # Build route từ spawn point
    self.route = self._build_route(spawn_point)
    self.current_waypoint_index = 0
    
    self.current_spawn_index += 1
    return self.vehicle
```

#### 3. `_build_route(spawn_point)`
Xây dựng route từ spawn point.

```python
def _build_route(self, start_transform, route_length=500):
    route = []
    current_location = start_transform.location
    current_waypoint = self.map.get_waypoint(current_location)
    
    for _ in range(route_length // 2):
        route.append(current_waypoint)
        if current_waypoint.next(2.0):
            current_waypoint = random.choice(current_waypoint.next(2.0))
        else:
            break
    
    return route
```

#### 4. `setup_sensors()`
Cài đặt sensors cho vehicle.

```python
def setup_sensors(self):
    self.camera_sensor = CameraSensorRGBPPO(self.vehicle)
    self.collision_sensor = CollisionSensor(self.vehicle)
```

#### 5. `get_observation()`
Lấy observation từ environment.

```python
def get_observation(self):
    # Get RGB image từ camera
    rgb_image = self.camera_sensor.front_camera[0]  # 80x160x3
    
    # Get navigation state
    throttle = self.vehicle.get_control().throttle
    velocity = self.vehicle.get_velocity()
    speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
    steer = self.vehicle.get_control().steer
    
    # Calculate lateral distance và heading error
    lateral_distance, heading_error = self._calculate_navigation_errors()
    
    # Normalize
    navigation_state = np.array([
        throttle,
        min(speed / 30.0, 1.5),
        steer,
        np.clip(lateral_distance / 3.0, -1.0, 1.0),
        np.clip(heading_error / 20.0, -1.0, 1.0)
    ], dtype=np.float32)
    
    return rgb_image, navigation_state
```

#### 6. `_calculate_navigation_errors()`
Tính khoảng cách ngang và sai lệch heading.

```python
def _calculate_navigation_errors(self):
    vehicle_location = self.vehicle.get_location()
    vehicle_rotation = self.vehicle.get_rotation()
    
    # Get current waypoint
    current_waypoint = self.map.get_waypoint(vehicle_location)
    
    # Get route waypoint
    if self.current_waypoint_index < len(self.route):
        route_waypoint = self.route[self.current_waypoint_index]
    else:
        route_waypoint = current_waypoint
    
    # Calculate signed lateral distance
    route_direction = route_waypoint.transform.get_forward_vector()
    route_direction = np.array([route_direction.x, route_direction.y])
    route_length = np.linalg.norm(route_direction)
    
    vehicle_to_route = np.array([
        route_waypoint.transform.location.x - vehicle_location.x,
        route_waypoint.transform.location.y - vehicle_location.y
    ])
    
    cross_product = np.cross(route_direction, vehicle_to_route)
    signed_distance = cross_product / route_length
    
    # Calculate signed heading error
    vehicle_fwd = vehicle_rotation.get_forward_vector()
    vehicle_heading = np.degrees(np.arctan2(vehicle_fwd.y, vehicle_fwd.x))
    
    route_heading = np.degrees(np.arctan2(route_direction[1], route_direction[0]))
    
    heading_error = vehicle_heading - route_heading
    heading_error = (heading_error + 180) % 360 - 180  # Normalize to [-180, 180]
    
    return signed_distance, heading_error
```

#### 7. `step(action)`
Thực hiện action và trả về observation mới, reward, done.

```python
def step(self, action):
    # action: [steer, throttle]
    steer, throttle = action
    
    # Apply control
    self.vehicle.apply_control(
        carla.VehicleControl(
            steer=float(steer),
            throttle=float(throttle),
            brake=0.0,
            hand_brake=False,
            reverse=False
        )
    )
    
    # Tick simulation
    self.world.tick()
    
    # Get new observation
    rgb_image, navigation_state = self.get_observation()
    
    # Calculate reward
    reward = self.calculate_reward()
    
    # Check done conditions
    done = self.check_done()
    
    # Update stuck counter
    self._update_stuck_counter()
    
    return (rgb_image, navigation_state), reward, done, {}
```

#### 8. `calculate_reward()`
Tính reward dựa trên driving performance.

```python
def calculate_reward(self):
    vehicle_location = self.vehicle.get_location()
    velocity = self.vehicle.get_velocity()
    speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
    
    # Get distance from center và heading error
    distance_from_center, heading_error = self._calculate_navigation_errors()
    
    # Factors
    centering_factor = max(1.0 - abs(distance_from_center) / 3.0, 0.0)
    angle_factor = max(1.0 - abs(heading_error) / 20.0, 0.0)
    
    # Speed factor
    target_speed = 20  # km/h
    min_speed = 5
    max_speed = 30
    
    if speed < min_speed:
        speed_factor = speed / min_speed
    elif speed <= target_speed:
        speed_factor = 1.0
    else:
        speed_factor = max(1.0 - (speed - target_speed) / (max_speed - target_speed), 0.0)
    
    # Base reward
    reward = 1.2 * speed_factor * centering_factor * angle_factor
    
    # Heading bonus
    if abs(heading_error) < 5:
        reward += 0.4
    elif abs(heading_error) < 10:
        reward += 0.2
    
    # Lane deviation penalties
    if distance_from_center > 2.5:
        reward -= 2.0
    elif distance_from_center > 2.0:
        reward -= 1.0
    elif distance_from_center > 1.5:
        reward -= 0.5
    
    # Terminal rewards
    if self.collision_sensor.collision:
        reward = -10
    
    if distance_from_center > 3.0:
        reward = -10  # Off road
    
    if self.stuck_counter > 100:
        reward = -10  # Stuck
    
    if speed > 33:
        reward = -5  # Extreme overspeed
    
    # Route completion bonus
    if self.current_waypoint_index >= len(self.route) - 1:
        reward += 10.0
    
    return reward
```

#### 9. `check_done()`
Kiểm tra điều kiện kết thúc episode.

```python
def check_done(self):
    if self.collision_sensor.collision:
        return True
    
    distance_from_center, _ = self._calculate_navigation_errors()
    if distance_from_center > 3.0:
        return True
    
    if self.stuck_counter > 100:
        return True
    
    if self.current_waypoint_index >= len(self.route) - 1:
        return True
    
    return False
```

#### 10. `reset()`
Reset environment cho episode mới.

```python
def reset(self):
    # Destroy all actors
    self.destroy()
    
    # Reconnect
    self.create_carla_connection()
    
    # Spawn new vehicle
    self.spawn_vehicle()
    
    # Setup sensors
    self.setup_sensors()
    
    # Reset counters
    self.stuck_counter = 0
    self.previous_location = self.vehicle.get_location()
    
    # Get initial observation
    return self.get_observation()
```

#### 11. `destroy()`
Cleanup environment.

```python
def destroy(self):
    for actor in self.spawned_actors:
        if actor is not None and actor.is_alive:
            actor.destroy()
    self.spawned_actors.clear()
    
    if self.client:
        self.client.reload_world()
```

## Safe Spawn System

### Danh sách Spawn Points An toàn

```python
self.safe_spawn_points = [1, 4, 6, 7, 8, 9, 10, 11]
```

### Round-Robin Strategy

```python
# Lần lượt dùng từng spawn point
spawn_point_index = self.safe_spawn_points[self.current_spawn_index]
self.current_spawn_index += 1

# Khi hết danh sách, shuffle và bắt đầu lại
if self.current_spawn_index >= len(self.safe_spawn_points):
    self.current_spawn_index = 0
    random.shuffle(self.safe_spawn_points)
```

## Stuck Detection

### Algorithm

```python
def _update_stuck_counter(self):
    current_location = self.vehicle.get_location()
    
    if self.previous_location is None:
        self.previous_location = current_location
        return
    
    # Calculate distance moved
    distance = np.sqrt(
        (current_location.x - self.previous_location.x) ** 2 +
        (current_location.y - self.previous_location.y) ** 2
    )
    
    # Increment counter if not moving
    if distance < 0.1:
        self.stuck_counter += 1
    else:
        self.stuck_counter = 0
    
    self.previous_location = current_location
```

## Environment Settings

File: `simulation/settings.py`

```python
# Environment settings
TARGET_SPEED = 20  # km/h
MIN_SPEED = 5
MAX_SPEED = 30
LANE_LIMIT = 3.0  # meters
HEADING_LIMIT = 20.0  # degrees

# Terminal conditions
COLLISION_REWARD = -10
OFF_ROAD_REWARD = -10
STUCK_THRESHOLD = 100  # steps
OVERSPEED_THRESHOLD = 33  # km/h
OVERSPEED_REWARD = -5

# Route completion
ROUTE_COMPLETION_REWARD = 10.0
```

## Best Practices

### Training Stability
- Dùng safe spawn points để tránh spawn không hợp lệ
- Reset environment sau mỗi episode
- Clean up actors đúng cách

### Performance
- Tick simulation thay vì wait_for_tick
- Giảm số lượng sensors không cần thiết
- Dùng synchronous mode

### Debugging
- Log navigation state để debug
- Visualize route và vehicle position
- Monitor stuck counter

## Next Steps

- [08_VAE_RGB.md](08_VAE_RGB.md) - VAE processing
- [09_PPO.md](09_PPO.md) - PPO training
- [11_Reward_Control.md](11_Reward_Control.md) - Reward tuning
