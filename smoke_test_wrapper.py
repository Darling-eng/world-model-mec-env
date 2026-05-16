from __future__ import annotations

from sim import GymnasiumMECEnv


def run_once(action_mode: str) -> None:
    env = GymnasiumMECEnv(action_mode=action_mode)
    obs, info = env.reset(seed=7)
    print(
        f"mode={action_mode} obs_shape={obs.shape} action_shape={env.action_space.shape} "
        f"step={info['step']}"
    )

    for index in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"t={index + 1} reward={reward:.3f} terminated={terminated} truncated={truncated} "
            f"accepted={info['accepted_action']} binary={info['binary_action']}"
        )
    env.close()


def main() -> None:
    for action_mode in ("multibinary", "box"):
        run_once(action_mode)


if __name__ == "__main__":
    main()
