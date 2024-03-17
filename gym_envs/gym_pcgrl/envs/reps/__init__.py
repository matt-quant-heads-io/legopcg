from gym_envs.gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from gym_envs.gym_pcgrl.envs.reps.narrow_cast_rep import NarrowCastRepresentation
from gym_envs.gym_pcgrl.envs.reps.narrow_multi_rep import NarrowMultiRepresentation
from gym_envs.gym_pcgrl.envs.reps.wide_rep import WideRepresentation
from gym_envs.gym_pcgrl.envs.reps.turtle_rep import TurtleRepresentation
from gym_envs.gym_pcgrl.envs.reps.turtle_cast_rep import TurtleCastRepresentation
from gym_envs.gym_pcgrl.envs.reps.turtle_rep_3d import TurtleRepresntation3D
from gym_envs.gym_pcgrl.envs.reps.wide_rep_3d import WideRepresentation3D
from gym_envs.gym_pcgrl.envs.reps.narrow_rep_3d import NarrowRepresentation3D

# all the representations should be defined here with its corresponding class
REPRESENTATIONS = {
    "narrow": NarrowRepresentation,
    "narrowcast": NarrowCastRepresentation,
    "narrowmulti": NarrowMultiRepresentation,
    "wide": WideRepresentation,
    "turtle": TurtleRepresentation,
    "turtlecast": TurtleCastRepresentation,
    "turtle3d": TurtleRepresntation3D,
    "wide3d": WideRepresentation3D,
    "narrow3d": NarrowRepresentation3D,
}
