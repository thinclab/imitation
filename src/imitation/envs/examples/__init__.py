"""Environments used for testing and benchmarking.

These are not a core part of the imitation package. They are relatively lightly tested,
and may be changed without warning.
"""

# Register environments with Gym
from imitation.envs.examples import model_envs  # noqa: F401


from typing import Optional

from gym.envs import register as gym_register

_ENTRY_POINT_PREFIX = "imitation.envs.airl_envs"


def _register(env_name: str, entry_point: str, kwargs: Optional[dict] = None):
    entry_point = f"{_ENTRY_POINT_PREFIX}.{entry_point}"
    gym_register(id=env_name, entry_point=entry_point, kwargs=kwargs)

_register("imitationNM/SortingOnions-v0", entry_point="sorting_onions_env:SortingOnions") 

_register("imitationNM/PatrolModel-v0", entry_point="perimeter_patrol_env:PatrolModel") 

_register("imitationNM/DiscretizedStateMountainCar-v0", entry_point="discrtzd_mountain_car:DiscretizedStateMountainCarEnv") 
