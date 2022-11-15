from hashlib import md5
import subprocess
from l2r import build_env
from l2r import RacingEnv
from src.config.yamlize import NameToSourcePath, create_configurable
import sys
import logging


if __name__ == "__main__":
    # Build environment
    env = build_env(
        controller_kwargs={"quiet": True},
        env_kwargs={
            "multimodal": True,
            "eval_mode": True,
            "n_eval_laps": 5,
            "max_timesteps": 5000,
            "obs_delay": 0.1,
            "not_moving_timeout": 50000,
            "reward_pol": "custom",
            "provide_waypoints": False,
            "active_sensors": ["CameraFrontRGB"],
            "vehicle_params": False,
        },
        action_cfg={
            "ip": "0.0.0.0",
            "port": 7077,
            "max_steer": 0.3,
            "min_steer": -0.3,
            "max_accel": 6,
            "min_accel": -1,
        },
    )
    runner = create_configurable(
        "config_files/example_sac/runner.yaml", NameToSourcePath.runner
    )

    with open(
        f"{runner.agent.model_save_path}/{runner.exp_config['experiment_name']}/git_config",
        "w+",
    ) as f:
        f.write(" ".join(sys.argv[1:3]))
    # Race!
    try:
        runner.run(env, sys.argv[3])
    except IndexError as e:
        logging.warning(e)
        runner.run(env, "")
