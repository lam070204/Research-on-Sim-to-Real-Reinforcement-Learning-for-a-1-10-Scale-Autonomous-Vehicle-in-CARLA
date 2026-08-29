# -*- coding: utf-8 -*-
# One-time GĐ6 integration patcher.
#
# Run from project root:
#   python .\apply_gd6_integration.py
#
# Patches:
#   simulation\carla_environment_rgb_v3.py
#   train_ppo_rgb_v3.py
#
# Backup:
#   dynamics_calibration\backups\gd6_integration\

from __future__ import print_function

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / "simulation" / "carla_environment_rgb_v3.py"
TRAIN_PATH = ROOT / "train_ppo_rgb_v3.py"
BACKUP_DIR = (
    ROOT
    / "dynamics_calibration"
    / "backups"
    / "gd6_integration"
)

ENV_MARKER = "# GĐ6_DOMAIN_RANDOMIZATION_INTEGRATED"
TRAIN_MARKER = "# GĐ6_TRAIN_DOMAIN_RANDOMIZATION_INTEGRATED"


def require(text, needle, label):
    if needle not in text:
        raise RuntimeError(
            "Không tìm thấy anchor {}:\n{}".format(label, needle)
        )


def patch_env(text):
    if ENV_MARKER in text:
        print("ENV đã tích hợp GĐ6, bỏ qua patch lại.")
        return text

    import_anchor = "from action_controller_v3 import ActionControllerV3\n"
    require(text, import_anchor, "env import")

    imports = (
        import_anchor
        + "\n"
        + ENV_MARKER
        + "\n"
        + "from domain_randomization_v3 import (\n"
        + "    nominal_domain_parameters,\n"
        + "    sample_domain_randomization,\n"
        + ")\n"
        + "from domain_randomization_runtime_v3 import (\n"
        + "    apply_domain_physics_v3,\n"
        + "    DomainRandomizedActionControllerV3,\n"
        + ")\n"
    )
    text = text.replace(import_anchor, imports, 1)

    signature_anchor = (
        "        steer_rate_limit_per_s=None,\n"
        "        speed_rate_limit_mps2=None,\n"
        "    ):\n"
    )
    require(text, signature_anchor, "env __init__ signature")

    signature_new = (
        "        steer_rate_limit_per_s=None,\n"
        "        speed_rate_limit_mps2=None,\n"
        "        domain_randomization_enabled=False,\n"
        "        domain_randomization_seed=None,\n"
        "    ):\n"
    )
    text = text.replace(signature_anchor, signature_new, 1)

    attrs_anchor = (
        "        self.steer_rate_limit_per_s = steer_rate_limit_per_s\n"
        "        self.speed_rate_limit_mps2 = speed_rate_limit_mps2\n"
    )
    require(text, attrs_anchor, "env attributes")

    attrs_new = (
        attrs_anchor
        + "\n"
        + "        self.domain_randomization_enabled = bool(\n"
        + "            domain_randomization_enabled\n"
        + "        )\n"
        + "        self.domain_randomization_seed = domain_randomization_seed\n"
        + "        self._domain_rng = random.Random(domain_randomization_seed)\n"
        + "        self.domain_randomization_params = nominal_domain_parameters()\n"
        + "        self.domain_randomization_applied = None\n"
    )
    text = text.replace(attrs_anchor, attrs_new, 1)

    method_anchor = "    def _settle_vehicle(self):\n"
    require(text, method_anchor, "before _settle_vehicle")

    new_methods = """    def _prepare_episode_domain_v3(self):
        if self.domain_randomization_enabled:
            params = sample_domain_randomization(
                rng=self._domain_rng
            )
        else:
            params = nominal_domain_parameters()

        self.domain_randomization_params = dict(params)

        self.domain_randomization_applied = apply_domain_physics_v3(
            vehicle=self.vehicle,
            params=self.domain_randomization_params,
            carla=carla,
        )

        self.last_sim_frame = int(self.world.tick())

        verify = self.vehicle.get_physics_control()

        expected_brake = float(
            self.domain_randomization_params["max_brake_torque"]
        )
        applied_wheels = list(verify.wheels)

        brake_ok = all(
            abs(float(w.max_brake_torque) - expected_brake) < 1e-3
            for w in applied_wheels
        )

        autobox_ok = (
            bool(verify.use_gear_autobox)
            == bool(
                self.domain_randomization_params[
                    "use_gear_autobox"
                ]
            )
        )

        gear_switch_ok = (
            abs(
                float(verify.gear_switch_time)
                - float(
                    self.domain_randomization_params[
                        "gear_switch_time_s"
                    ]
                )
            )
            < 1e-6
        )

        if not brake_ok or not autobox_ok or not gear_switch_ok:
            raise RuntimeError(
                "GĐ6 PhysicsControl readback mismatch."
            )

        print(
            "GĐ6 DOMAIN | random={} | torque_scale={:.4f} | "
            "brake={:.1f} | steer_gain(+/-)={:.4f}/{:.4f} | "
            "delay={} tick".format(
                self.domain_randomization_enabled,
                float(
                    self.domain_randomization_params[
                        "torque_scale"
                    ]
                ),
                float(
                    self.domain_randomization_params[
                        "max_brake_torque"
                    ]
                ),
                float(
                    self.domain_randomization_params[
                        "steer_gain_positive"
                    ]
                ),
                float(
                    self.domain_randomization_params[
                        "steer_gain_negative"
                    ]
                ),
                int(
                    self.domain_randomization_params[
                        "extra_command_delay_ticks"
                    ]
                ),
            )
        )

"""
    text = text.replace(
        method_anchor,
        new_methods + method_anchor,
        1,
    )

    controller_old = """        self.action_controller = ActionControllerV3(
            vehicle=self.vehicle,
            steer_rate_limit_per_s=self.steer_rate_limit_per_s,
            speed_rate_limit_mps2=self.speed_rate_limit_mps2,
        )
        self.action_controller.reset()
"""
    require(text, controller_old, "ActionController block")

    controller_new = """        base_action_controller = ActionControllerV3(
            vehicle=self.vehicle,
            steer_rate_limit_per_s=self.steer_rate_limit_per_s,
            speed_rate_limit_mps2=self.speed_rate_limit_mps2,
        )

        self.action_controller = DomainRandomizedActionControllerV3(
            base_controller=base_action_controller,
            steer_gain_positive=float(
                self.domain_randomization_params[
                    "steer_gain_positive"
                ]
            ),
            steer_gain_negative=float(
                self.domain_randomization_params[
                    "steer_gain_negative"
                ]
            ),
            extra_command_delay_ticks=int(
                self.domain_randomization_params[
                    "extra_command_delay_ticks"
                ]
            ),
        )
        self.action_controller.reset()
"""
    text = text.replace(controller_old, controller_new, 1)

    reset_anchor = (
        "            self._apply_vehicle_geometry()\n"
        "            self._settle_vehicle()\n"
    )
    require(text, reset_anchor, "reset domain insertion")

    reset_new = (
        "            self._apply_vehicle_geometry()\n"
        "            self._prepare_episode_domain_v3()\n"
        "            self._settle_vehicle()\n"
    )
    text = text.replace(reset_anchor, reset_new, 1)

    info_anchor = '            info["imu"] = imu_state\n'
    require(text, info_anchor, "step info")

    info_new = (
        info_anchor
        + '            info["domain_randomization"] = dict(\n'
        + "                self.domain_randomization_params\n"
        + "            )\n"
    )
    text = text.replace(info_anchor, info_new, 1)

    return text


def patch_train(text):
    if TRAIN_MARKER in text:
        print("TRAIN đã tích hợp GĐ6, bỏ qua patch lại.")
        return text

    anchor = (
        "            encoder_device=args.encoder_device,\n"
        "        )\n"
    )
    require(text, anchor, "trainer env constructor")

    new = (
        "            encoder_device=args.encoder_device,\n"
        + "            "
        + TRAIN_MARKER
        + "\n"
        + "            domain_randomization_enabled=bool(args.train),\n"
        + "            domain_randomization_seed=int(args.seed),\n"
        + "        )\n"
    )

    return text.replace(anchor, new, 1)


def main():
    if not ENV_PATH.is_file():
        raise FileNotFoundError(str(ENV_PATH))
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(str(TRAIN_PATH))

    runtime = ROOT / "domain_randomization_runtime_v3.py"
    config = ROOT / "domain_randomization_v3.py"

    if not runtime.is_file():
        raise FileNotFoundError(
            "Thiếu domain_randomization_runtime_v3.py"
        )
    if not config.is_file():
        raise FileNotFoundError(
            "Thiếu domain_randomization_v3.py"
        )

    env_text = ENV_PATH.read_text(encoding="utf-8")
    train_text = TRAIN_PATH.read_text(encoding="utf-8")

    new_env = patch_env(env_text)
    new_train = patch_train(train_text)

    compile(new_env, str(ENV_PATH), "exec")
    compile(new_train, str(TRAIN_PATH), "exec")

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    env_backup = BACKUP_DIR / "carla_environment_rgb_v3_before_gd6.py"
    train_backup = BACKUP_DIR / "train_ppo_rgb_v3_before_gd6.py"

    if not env_backup.exists():
        shutil.copy2(str(ENV_PATH), str(env_backup))
    if not train_backup.exists():
        shutil.copy2(str(TRAIN_PATH), str(train_backup))

    ENV_PATH.write_text(new_env, encoding="utf-8")
    TRAIN_PATH.write_text(new_train, encoding="utf-8")

    print("=" * 88)
    print("GĐ6 INTEGRATION PATCH APPLIED")
    print("=" * 88)
    print("ENV   :", ENV_PATH)
    print("TRAIN :", TRAIN_PATH)
    print("BACKUP:", BACKUP_DIR)
    print("Syntax validation: PASS")
    print("")
    print("NEXT:")
    print("  python -m py_compile .\\domain_randomization_runtime_v3.py")
    print("  python -m py_compile .\\simulation\\carla_environment_rgb_v3.py")
    print("  python -m py_compile .\\train_ppo_rgb_v3.py")
    print("  python .\\smoke_test_gd6_environment_v3.py")
    print("=" * 88)


if __name__ == "__main__":
    main()
