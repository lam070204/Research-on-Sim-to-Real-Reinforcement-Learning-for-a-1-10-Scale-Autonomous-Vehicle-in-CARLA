# 06 Sensors

## Hệ thống Sensor

Tài liệu về các sensor được sử dụng trong dự án Autonomous Driving.

## Tổng quan

Dự án sử dụng các sensor sau để thu thập thông tin từ môi trường CARLA:

| Sensor | Mục đích | Frequency |
|--------|----------|-----------|
| RGB Camera | Nhận diện đường, môi trường | 20 Hz |
| Collision Sensor | Phát hiện va chạm | Event-based |
| GNSS | Vị trí toàn cầu | 10 Hz |
| IMU | Gia tốc, vận tốc góc | 100 Hz |

## RGB Camera

### Thông số Kỹ thuật

| Thông số | Giá trị |
|----------|---------|
| Độ phân giải | 160x80 pixels |
| FOV | 90° |
| FPS | 20 |
| Vị trí X | 2.5m (trước xe) |
| Vị trí Y | 0.0m (giữa xe) |
| Vị trí Z | 1.5m (độ cao) |
| Pitch | 0.0° |
| Yaw | 0.0° |
| Roll | 0.0° |

### Cài đặt trong Code

File: `simulation/sensors.py`

```python
# Camera settings
FRONT_CAMERA_WIDTH = 160
FRONT_CAMERA_HEIGHT = 80
FRONT_CAMERA_FPS = 20
FRONT_CAMERA_FOV = 90

# Camera position relative to vehicle
FRONT_CAMERA_X = 2.5   # Forward
FRONT_CAMERA_Y = 0.0   # Center
FRONT_CAMERA_Z = 1.5   # Height

# Camera rotation
FRONT_CAMERA_PITCH = 0.0
FRONT_CAMERA_YAW = 0.0
FRONT_CAMERA_ROLL = 0.0

# Blueprint name
RGB_CAMERA = "sensor.camera.rgb"
```

### Class CameraSensorRGBPPO

File: `simulation/environment_rgb_v2.py`

```python
class CameraSensorRGBPPO:
    def __init__(self, vehicle):
        self.sensor_name = RGB_CAMERA
        self.parent = vehicle
        self.front_camera = []
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensorRGBPPO._get_rgb_data(weak_self, image)
        )

    def _set_camera_sensor(self, world):
        camera_bp = world.get_blueprint_library().find(self.sensor_name)
        camera_bp.set_attribute("image_size_x", str(FRONT_CAMERA_WIDTH))
        camera_bp.set_attribute("image_size_y", str(FRONT_CAMERA_HEIGHT))
        camera_bp.set_attribute("fov", str(FRONT_CAMERA_FOV))
        camera_bp.set_attribute("sensor_tick", str(1.0 / FRONT_CAMERA_FPS))

        if camera_bp.has_attribute("enable_postprocess_effects"):
            camera_bp.set_attribute("enable_postprocess_effects", "true")
        if camera_bp.has_attribute("gamma"):
            camera_bp.set_attribute("gamma", "2.2")
        if camera_bp.has_attribute("exposure_mode"):
            camera_bp.set_attribute("exposure_mode", "histogram")

        transform = carla.Transform(
            carla.Location(
                x=FRONT_CAMERA_X, y=FRONT_CAMERA_Y, z=FRONT_CAMERA_Z
            ),
            carla.Rotation(
                pitch=FRONT_CAMERA_PITCH,
                yaw=FRONT_CAMERA_YAW,
                roll=FRONT_CAMERA_ROLL,
            ),
        )
        return world.spawn_actor(
            camera_bp,
            transform,
            attach_to=self.parent,
            attachment_type=carla.AttachmentType.Rigid,
        )

    @staticmethod
    def _get_rgb_data(weak_self, image):
        self = weak_self()
        if self is None:
            return
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        rgb_image = array[:, :, :3][:, :, ::-1].copy()
        self.front_camera.clear()
        self.front_camera.append(rgb_image)
```

### Xử lý Ảnh

```python
# 1. Get raw data từ CARLA
raw_data = image.raw_data  # BGRA format

# 2. Convert to numpy array
array = np.frombuffer(raw_data, dtype=np.uint8)
array = array.reshape((image.height, image.width, 4))

# 3. Convert BGRA to RGB
rgb_image = array[:, :, :3][:, :, ::-1].copy()

# 4. Normalize về [0, 1]
normalized = rgb_image.astype(np.float32) / 255.0

# 5. Transpose cho PyTorch (C, H, W)
tensor = torch.from_numpy(normalized).permute(2, 0, 1)
```

## Collision Sensor

### Mục đích
Phát hiện va chạm để kết thúc episode và áp dụng negative reward.

### Cài đặt

```python
class CollisionSensor:
    def __init__(self, vehicle):
        self.sensor_name = "sensor.other.collision"
        self.parent = vehicle
        self.collision = False
        world = self.parent.get_world()
        self.sensor = self._set_collision_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event)
        )

    def _set_collision_sensor(self, world):
        collision_bp = world.get_blueprint_library().find(self.sensor_name)
        return world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.parent,
            attachment_type=carla.AttachmentType.Rigid,
        )

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if self is None:
            return
        self.collision = True
```

### Xử lý Collision

```python
# Check collision status
if collision_sensor.collision:
    reward = -10  # Negative reward
    done = True   # End episode
    collision_sensor.collision = False  # Reset
```

## GNSS Sensor

### Mục đích
Cung cấp vị trí toàn cầu (latitude, longitude, altitude).

### Cài đặt

```python
class GNSSSensor:
    def __init__(self, vehicle):
        self.sensor_name = "sensor.other.gnss"
        self.parent = vehicle
        self.gnss_data = None
        world = self.parent.get_world()
        self.sensor = self._set_gnss_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: GNSSSensor._on_gnss_event(weak_self, event)
        )

    def _set_gnss_sensor(self, world):
        gnss_bp = world.get_blueprint_library().find(self.sensor_name)
        gnss_bp.set_attribute("sensor_tick", "0.1")  # 10 Hz
        return world.spawn_actor(
            gnss_bp,
            carla.Transform(carla.Location(x=1.0, z=2.8)),
            attach_to=self.parent,
            attachment_type=carla.AttachmentType.Rigid,
        )

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if self is None:
            return
        self.gnss_data = (event.latitude, event.longitude, event.altitude)
```

## IMU Sensor

### Mục đích
Đo gia tốc tuyến tính và vận tc góc.

### Cài đặt

```python
class IMUSensor:
    def __init__(self, vehicle):
        self.sensor_name = "sensor.other.imu"
        self.parent = vehicle
        self.imu_data = None
        world = self.parent.get_world()
        self.sensor = self._set_imu_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: IMUSensor._on_imu_event(weak_self, event)
        )

    def _set_imu_sensor(self, world):
        imu_bp = world.get_blueprint_library().find(self.sensor_name)
        imu_bp.set_attribute("sensor_tick", "0.01")  # 100 Hz
        return world.spawn_actor(
            imu_bp,
            carla.Transform(carla.Location(x=1.0, z=2.8)),
            attach_to=self.parent,
            attachment_type=carla.AttachmentType.Rigid,
        )

    @staticmethod
    def _on_imu_event(weak_self, event):
        self = weak_self()
        if self is None:
            return
        self.imu_data = {
            'accelerometer': (event.accelerometer.x, event.accelerometer.y, event.accelerometer.z),
            'gyroscope': (event.gyroscope.x, event.gyroscope.y, event.gyroscope.z),
            'compass': event.compass
        }
```

## Sensor Configuration File

File: `simulation/settings.py`

```python
# Sensor settings
RGB_CAMERA = "sensor.camera.rgb"
COLLISION_SENSOR = "sensor.other.collision"
GNSS_SENSOR = "sensor.other.gnss"
IMU_SENSOR = "sensor.other.imu"

# Camera parameters
FRONT_CAMERA_WIDTH = 160
FRONT_CAMERA_HEIGHT = 80
FRONT_CAMERA_FPS = 20
FRONT_CAMERA_FOV = 90

# Camera position
FRONT_CAMERA_X = 2.5
FRONT_CAMERA_Y = 0.0
FRONT_CAMERA_Z = 1.5

# Camera rotation
FRONT_CAMERA_PITCH = 0.0
FRONT_CAMERA_YAW = 0.0
FRONT_CAMERA_ROLL = 0.0
```

## Best Practices

### Performance
- Giảm độ phân giải camera xuống 160x80 để giảm computation
- Dùng 20 FPS đủ cho RL training
- Chỉ listen sensor khi cần thiết

### Stability
- Dùng weakref để tránh memory leak
- Reset sensor state sau mỗi episode
- Handle exceptions khi sensor fails

### Data Quality
- Calibrate camera position cho góc nhìn tốt nhất
- Dùng Raw color converter cho ảnh chính xác
- Sync sensor data với simulation tick

## Troubleshooting

### Lỗi: "Sensor not found"
- Kiểm tra blueprint name chính xác
- Verify CARLA version compatibility

### Lỗi: "Sensor data không update"
- Kiểm tra synchronous mode settings
- Verify sensor_tick attribute

### Lỗi: "Memory leak"
- Dùng weakref cho callbacks
- Destroy sensors khi không dùng

## Next Steps

- [07_Environment.md](07_Environment.md) - Environment setup
- [08_VAE_RGB.md](08_VAE_RGB.md) - VAE processing
- [09_PPO.md](09_PPO.md) - PPO training
