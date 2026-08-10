# 04 RoadRunner Map

## Tạo Map Tùy chỉnh với RoadRunner

Hướng dẫn tạo map tùy chỉnh cho CARLA sử dụng RoadRunner của MathWorks.

## Giới thiệu

RoadRunner là công cụ tạo map 3D cho autonomous driving simulation, hỗ trợ xuất ra định dạng OpenDRIVE (.xodr) và FBX (.fbx) để sử dụng trong CARLA.

## Cài đặt RoadRunner

### Yêu cầu
- MATLAB R2020b trở lên
- RoadRunner Toolbox
- GPU hỗ trợ DirectX 11/12

### Các bước cài đặt
1. Mở MATLAB
2. Vào Add-Ons → Get Add-Ons
3. Tìm và cài đặt "RoadRunner Toolbox"
4. Khởi động RoadRunner từ MATLAB command:
```matlab
roadrunner
```

## Tạo Map Mới

### Bước 1: Tạo Project
1. Mở RoadRunner
2. File → New Project
3. Chọn thư mục lưu project
4. Đặt tên project (ví dụ: `Town07_Custom`)

### Bước 2: Thiết lập Map Properties
1. Vào Project Settings
2. Thiết lập:
   - **Map Size**: 500m x 500m (tùy nhu cầu)
   - **Tile Size**: 100m
   - **Coordinate System**: CARLA compatible

### Bước 3: Thiết kế Road Network

#### Tạo Road Segment
1. Chọn công cụ "Road" từ toolbar
2. Click để đặt điểm bắt đầu
3. Click thêm điểm để tạo đường cong/thẳng
4. Double-click để kết thúc segment

#### Thiết lập Lane
1. Chọn road segment
2. Trong Properties panel:
   - **Number of Lanes**: 1-4 lanes
   - **Lane Width**: 3.0-3.5m (tiêu chuẩn)
   - **Shoulder**: Có/không có lề

#### Tạo Giao lộ (Intersection)
1. Dùng công cụ "Intersection"
2. Chọn loại giao lộ:
   - 4-way intersection
   - T-junction
   - Roundabout
3. Kết nối các road segments

### Bước 4: Thêm Road Markings và Signs

#### Road Markings
1. Chọn "Lane Markings" tool
2. Chọn loại marking:
   - Solid line (vạch liền)
   - Dashed line (vạch đứt)
   - Double yellow line
3. Apply lên lane edges

#### Traffic Signs
1. Chọn "Traffic Signs" tool
2. Drag & drop signs từ library:
   - Stop sign
   - Speed limit
   - Yield sign
   - Traffic light

### Bước 5: Export cho CARLA

#### Export OpenDRIVE
1. File → Export → OpenDRIVE
2. Chọn thư mục export
3. File sẽ có định dạng `.xodr`

#### Export FBX (3D Mesh)
1. File → Export → FBX
2. Settings:
   - **Scale**: 1.0 (meters)
   - **Include Textures**: Yes
   - **LOD**: Medium
3. Export

## Import Map vào CARLA

### Bước 1: Chuẩn bị File
```
Import/
├── Town07_Custom.xodr      # OpenDRIVE file
└── Town07_Custom.fbx       # 3D mesh
```

### Bước 2: Copy vào CARLA
```bash
# Copy OpenDRIVE file
xcopy "Town07_Custom.xodr" "C:\CARLA_0.9.13\CarlaUE4\Content\Carla\OpenDrive\Town07_Custom.xodr" /Y

# Copy FBX file
xcopy "Town07_Custom.fbx" "C:\CARLA_0.9.13\CarlaUE4\Content\Carla\Maps\Town07_Custom.fbx" /Y
```

### Bước 3: Import trong Unreal Engine
1. Mở Unreal Engine project CARLA
2. Import FBX file vào `Content/Carla/Maps/`
3. Build lighting
4. Save package

### Bước 4: Test Map
```bash
cd C:\CARLA_0.9.13
CarlaUE4.exe -map=Town07_Custom
```

## Map Đơn giản cho Training

### Đặc điểm Map Training Tốt
- **Đoạn thẳng dài**: 200-500m để test tốc độ
- **Ít giao lộ**: Giảm complexity khi training ban đầu
- **Lane rộng**: 3.5m để dễ keep lane
- **Không có traffic**: Tập trung vào driving cơ bản

### Gợi ý Spawn Points
```
Spawn 1: Start of straight segment
Spawn 4: Middle of straight segment  
Spawn 6-11: Various positions on straight/curve
```

## Troubleshooting

### Lỗi: Map không load được trong CARLA
- Kiểm tra OpenDRIVE syntax
- Verify FBX scale và coordinate system
- Rebuild Unreal Engine project

### Lỗi: Vehicle spawn không đúng vị trí
- Kiểm tra spawn points trong OpenDRIVE file
- Đảm bảo road có lane width hợp lệ

### Lỗi: Road markings không hiển thị
- Export FBX với textures enabled
- Check material settings trong Unreal Engine

## Tham khảo

- [RoadRunner Documentation](https://www.mathworks.com/help/roadrunner/)
- [CARLA Map Import Guide](https://carla.readthedocs.io/en/latest/how_to_build_map/)
- [OpenDRIVE Format](https://www.asam.net/standards/detail/opendrive/)

## Next Steps

- [05_CARLA.md](05_CARLA.md) - CARLA Simulator
- [06_Sensors.md](06_Sensors.md) - Hệ thống sensor
- [07_Environment.md](07_Environment.md) - Environment setup
