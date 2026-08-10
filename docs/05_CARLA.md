# 05 CARLA Simulator

## CARLA Simulator

Hướng dẫn sử dụng CARLA Simulator cho dự án Autonomous Driving.

## Giới thiệu CARLA

CARLA (Car Learning to Act) là simulator mã nguồn mở cho autonomous driving research, được phát triển bởi Intel Labs, Toyota Research Institute, và Computer Vision Center.

### Tính năng chính
- **Môi trường 3D**: Đô thị với buildings, roads, traffic signs
- **Weather System**: Nắng, mưa, sương mù, hoàng hôn
- **Traffic System**: NPC vehicles, pedestrians
- **Sensors**: Camera, LiDAR, Radar, IMU, GNSS
- **API**: Python và C++ API cho customization

## Phiên bản

Dự án sử dụng **CARLA 0.9.13**

### Thay đổi so với 0.9.8
- Cải thiện physics engine
- Hỗ trợ semantic segmentation tốt hơn
- Bug fixes cho camera sensors
- Cải thiện performance

## Khởi động CARLA

### Khởi động Server
```bash
cd C:\CARLA_0.9.13
CarlaUE4.exe
```

### Khởi động với Settings
```bash
# Chế độ cửa sổ
CarlaUE4.exe -windowed

# Chế độ fullscreen
CarlaUE4.exe -fullscreen

# Chế độ OpenGL (nếu DirectX có vấn đề)
CarlaUE4.exe -opengl

# Chỉ định map
CarlaUE4.exe -map=Town07

# Không có UI
CarlaUE4.exe -no-rendering

# Quality settings
CarlaUE4.exe -QualityLevel=Low
CarlaUE4.exe -QualityLevel=Medium
CarlaUE4.exe -QualityLevel=Epic
```

## Kết nối Python Client

### Kết nối cơ bản
```python
import sys
sys.path.append('carla/carla-0.9.13-py3.8-win-amd64.egg')
import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
```

### Lấy Map và Spawn Points
```python
map = world.get_map()
spawn_points = map.get_spawn_points()

print(f"Map: {map.name}")
print(f"Number of spawn points: {len(spawn_points)}")

# Print tất cả spawn points
for i, spawn in enumerate(spawn_points):
    print(f"Spawn {i+1}: {spawn.location}")
```

### Thiết lập Weather
```python
# Weather presets
weather = carla.WeatherParameters.ClearNoon
weather = carla.WeatherParameters.CloudyNoon
weather = carla.WeatherParameters.WetNoon
weather = carla.WeatherParameters.HardRainNoon
weather = carla.WeatherParameters.SoftRainNoon
weather = carla.WeatherParameters.WetCloudyNoon

# Custom weather
weather = carla.WeatherParameters(
    cloudiness=30.0,
    precipitation=0.0,
    precipitation_deposits=0.0,
    wind_intensity=10.0,
    sun_azimuth_angle=180.0,
    sun_altitude_angle=45.0
)
world.set_weather(weather)
```

### Thiết lập Simulation Settings
```python
settings = world.get_settings()
settings.fixed_delta_seconds = 1.0 / 20.0  # 20 FPS
settings.synchronous_mode = True  # Sync mode cho RL
settings.no_rendering_mode = False  # Render để visualize
world.apply_settings(settings)
```

## Spawn Vehicle

### Spawn Vehicle từ Blueprint
```python
blueprint_library = world.get_blueprint_library()

# Lấy vehicle blueprint
vehicle_bp = blueprint_library.filter('vehicle.*')[0]

# Spawn tại vị trí chỉ định
spawn_point = spawn_points[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
```

### Vehicle Controls
```python
# Set steering và throttle
vehicle.apply_control(
    carla.VehicleControl(
        steer=0.0,      # -1.0 to 1.0
        throttle=0.5,   # 0.0 to 1.0
        brake=0.0,      # 0.0 to 1.0
        hand_brake=False,
        reverse=False,
        manual_gears=False
    )
)

# Get vehicle state
velocity = vehicle.get_velocity()
speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # km/h

location = vehicle.get_location()
rotation = vehicle.get_rotation()
```

## Maps có sẵn

| Map | Đặc điểm | Spawn Points |
|-----|----------|--------------|
| Town01 | Urban, nhiều giao lộ | 20+ |
| Town02 | Suburban, ít traffic | 20+ |
| Town03 | Industrial area | 20+ |
| Town04 | Highway, rural | 20+ |
| Town05 | Large urban | 20+ |
| Town06 | Rural, mountains | 20+ |
| Town07 | Countryside, simple | 12 |

### Map khuyến nghị cho Training
- **Town07**: Đơn giản, ít giao lộ, phù hợp training ban đầu
- **Town04**: Highway dài, tốt cho speed control
- **Custom Map**: Tạo với RoadRunner cho scenarios cụ thể

## API Reference

### Client
```python
client = carla.Client(host, port)
client.set_timeout(seconds)
world = client.get_world()
client.load_world(map_name)
client.reload_world()
```

### World
```python
world = client.get_world()
world.get_map()
world.get_blueprint_library()
world.get_spawn_points()
world.get_weather()
world.set_weather(weather)
world.get_settings()
world.apply_settings(settings)
world.spawn_actor(blueprint, transform)
world.tick()
world.wait_for_tick(seconds)
```

### Actor (Vehicle)
```python
actor.get_location()
actor.get_rotation()
actor.get_velocity()
actor.get_acceleration()
actor.get_angular_velocity()
actor.apply_control(control)
actor.destroy()
```

## Best Practices

### Performance
- Dùng synchronous mode cho RL training
- Set fixed_delta_seconds phù hợp với FPS mong muốn
- Giảm quality settings nếu cần FPS cao
- Dùng no_rendering_mode khi không cần visualize

### Stability
- Set timeout đủ lớn cho client
- Handle exceptions khi spawn actors
- Clean up actors khi kết thúc episode
- Avoid quá nhiều actors cùng lúc

## Troubleshooting

### Lỗi: "Failed to connect"
- Kiểm tra CARLA server đang chạy
- Verify port 2000 không bị block
- Tăng timeout: `client.set_timeout(20.0)`

### Lỗi: "Spawn failed"
- Kiểm tra spawn point không bị blocked
- Try spawn point khác
- Destroy vehicle cũ trước khi spawn mới

### Lỗi: "Simulation lags"
- Giảm quality settings
- Giảm số lượng NPCs
- Dùng synchronous mode

## Next Steps

- [06_Sensors.md](06_Sensors.md) - Hệ thống sensor
- [07_Environment.md](07_Environment.md) - Environment setup
- [09_PPO.md](09_PPO.md) - Training PPO
