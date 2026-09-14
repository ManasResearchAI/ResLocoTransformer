"""
envs/builder.py — Environment factory and vectorised-env builders.

keeping only the UnitreeMujocoGymEnv path.  All A1/pybullet code removed.
"""

import gymnasium as gym
import numpy as np

from torchrl.env.continuous_wrapper import NormAct, NormObs, NormRet, Normalizer
from torchrl.env.base_wrapper import BaseWrapper
from torchrl.env.vecenv import VecEnv
from torchrl.env.subproc_vecenv import SubProcVecEnv
from gymnasium.wrappers import TimeLimit


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_unitree_mujoco_env(
    scene_path: str,
    get_image: bool = True,
    **kwargs,
):
    from envs.mujoco_env import UnitreeMujocoGymEnv
    return UnitreeMujocoGymEnv(scene_path=scene_path, get_image=get_image, **kwargs)


ENV_DICT = {
    "UnitreeMujocoGymEnv": build_unitree_mujoco_env,
}

TIMELIMIT_DICT = {
    "UnitreeMujocoGymEnv": 1000,
}


# ---------------------------------------------------------------------------
# Observation normalizer with image pass-through
# ---------------------------------------------------------------------------

class NormObsWithImg(gym.ObservationWrapper, BaseWrapper):
    """
    Normalises only the proprioceptive portion of the observation.

    The observation is ``[proprio | gait_phase | image]``. Running mean/var is
    applied to the proprioceptive block only; the 2-dim gait phase
    ``[sin(phi), cos(phi)]`` and the depth block pass through untouched --
    the phase is already bounded in [-1, 1] and carries a fixed scale that
    whitening by running statistics would destroy.

    The split point is derived from the env constants rather than hard-coded,
    so it stays correct when STATE_DIM changes. (It was previously a literal
    84, which silently became "proprio only" when the gait phase pushed
    STATE_DIM to 86 -- right by luck, since the phase is appended last.)
    """

    def __init__(self, env, epsilon=1e-4, clipob=10.0):
        super().__init__(env)
        # Imported lazily, matching build_unitree_mujoco_env: keeps mujoco off
        # the import path until an env is actually constructed.
        from envs.mujoco_env import STATE_DIM, GAIT_PHASE_DIM

        self.training = True
        self.clipob = clipob
        self.state_shape = STATE_DIM - GAIT_PHASE_DIM
        self._obs_normalizer = Normalizer((self.state_shape,))

    def observation(self, observation):
        if self.training:
            self._obs_normalizer.update_estimate(
                observation[..., :self.state_shape]
            )
        img_obs = observation[..., self.state_shape:]
        state_obs = self._obs_normalizer.filt(observation[..., :self.state_shape])
        return np.hstack([
            state_obs,
            img_obs,
        ])


# ---------------------------------------------------------------------------
# Single-env builder
# ---------------------------------------------------------------------------

def make_env(env_name, env_build_params):
    return ENV_DICT[env_name](**env_build_params)


def get_single_env(env_id, env_param):
    env = make_env(env_id, env_param["env_build"])
    env = BaseWrapper(env)

    if "rew_norm" in env_param:
        env = NormRet(env, **env_param["rew_norm"])

    if env_id in TIMELIMIT_DICT:
        max_steps = env_param.get("horizon", TIMELIMIT_DICT[env_id])
        env = TimeLimit(env=env, max_episode_steps=max_steps)

    if isinstance(env.action_space, gym.spaces.Box):
        env = NormAct(env)

    return env


# ---------------------------------------------------------------------------
# Vectorised-env builders
# ---------------------------------------------------------------------------

def get_env(env_id, env_param):
    env = get_single_env(env_id, env_param)
    if env_param.get("obs_norm"):
        env = NormObsWithImg(env)
    return env


def get_vec_env(env_id, env_param, vec_env_nums):
    if isinstance(env_param, list):
        assert vec_env_nums % len(env_param) == 0
        env_args = [
            [env_id, sub] for sub in env_param
        ] * (vec_env_nums // len(env_param))
        vec_env = VecEnv(vec_env_nums, [get_single_env] * vec_env_nums, env_args)
        ref = env_param[0]
    else:
        vec_env = VecEnv(vec_env_nums, get_single_env, [env_id, env_param])
        ref = env_param

    if ref.get("obs_norm"):
        vec_env = NormObsWithImg(vec_env)
    return vec_env


def get_subprocvec_env(env_id, env_param, vec_env_nums, proc_nums):
    if isinstance(env_param, list):
        assert vec_env_nums % len(env_param) == 0
        env_args = [
            [env_id, sub] for sub in env_param
        ] * (vec_env_nums // len(env_param))
        vec_env = SubProcVecEnv(
            proc_nums, vec_env_nums, [get_single_env] * vec_env_nums, env_args
        )
        ref = env_param[0]
    else:
        vec_env = SubProcVecEnv(
            proc_nums, vec_env_nums, get_single_env, [env_id, env_param]
        )
        ref = env_param

    if ref.get("obs_norm"):
        vec_env = NormObsWithImg(vec_env)
    return vec_env
