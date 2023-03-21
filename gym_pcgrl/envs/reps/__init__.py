from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from gym_pcgrl.envs.reps.narrow_cast_rep import NarrowCastRepresentation
from gym_pcgrl.envs.reps.narrow_multi_rep import NarrowMultiRepresentation
from gym_pcgrl.envs.reps.wide_rep import WideRepresentation
from gym_pcgrl.envs.reps.turtle_rep import TurtleRepresentation
from gym_pcgrl.envs.reps.turtle_cast_rep import TurtleCastRepresentation
from gym_pcgrl.envs.reps.turtle_rep_3d import TurtleRepresntation3D
from gym_pcgrl.envs.reps.piecewise_lego_rep import PiecewiseRepresentation

# all the representations should be defined here with its corresponding class
REPRESENTATIONS = {
    "narrow": NarrowRepresentation,
    "narrowcast": NarrowCastRepresentation,
    "narrowmulti": NarrowMultiRepresentation,
    "wide": WideRepresentation,
    "turtle": TurtleRepresentation,
    "turtlecast": TurtleCastRepresentation,
    "turtle3d" : TurtleRepresntation3D,
    "piecewise" : PiecewiseRepresentation
}
