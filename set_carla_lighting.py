# -*- coding: utf-8 -*-
import argparse
from simulation.carla_connection_v3 import carla

PRESETS = [
    ("low_045_clear",   15.0,  45.0,  5.0),
    ("low_135_clear",   15.0, 135.0,  5.0),
    ("low_225_clear",   15.0, 225.0,  5.0),
    ("low_315_clear",   15.0, 315.0,  5.0),
    ("mid_045_clear",   30.0,  45.0, 10.0),
    ("mid_135_clear",   30.0, 135.0, 10.0),
    ("mid_225_clear",   30.0, 225.0, 10.0),
    ("mid_315_clear",   30.0, 315.0, 10.0),
    ("high_045_clear",  50.0,  45.0, 10.0),
    ("high_135_clear",  50.0, 135.0, 10.0),
    ("high_225_clear",  50.0, 225.0, 10.0),
    ("high_315_clear",  50.0, 315.0, 10.0),
    ("noon_soft",       70.0,  90.0, 35.0),
    ("noon_cloudy",     70.0, 270.0, 65.0),
    ("mid_cloudy_a",    35.0,  90.0, 55.0),
    ("mid_cloudy_b",    35.0, 270.0, 55.0),
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--preset", type=int, required=True)
    args = p.parse_args()

    idx = int(args.preset)
    if not (0 <= idx < len(PRESETS)):
        raise ValueError("--preset must be 0..15")

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()

    name, alt, az, cloud = PRESETS[idx]
    weather = world.get_weather()
    weather.sun_altitude_angle = float(alt)
    weather.sun_azimuth_angle = float(az)
    weather.cloudiness = float(cloud)
    weather.precipitation = 0.0
    weather.precipitation_deposits = 0.0
    weather.wetness = 0.0
    weather.fog_density = 0.0
    weather.fog_distance = 1000.0
    weather.wind_intensity = 0.0
    world.set_weather(weather)

    print(
        "SET WEATHER OK | preset={} | name={} | sun_alt={:.1f} | sun_az={:.1f} | cloud={:.1f}".format(
            idx, name, alt, az, cloud
        )
    )

if __name__ == "__main__":
    main()
