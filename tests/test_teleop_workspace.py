import numpy as np

from svla.sim import ArmSim
from svla.teleop_workspace import NULL_ARM_JOINT_POSITIONS, build_workspace_bounds, fk_ee_position


def test_null_configuration_fk_is_finite():
    sim = ArmSim()
    center = fk_ee_position(sim.model, sim.data, sim.controller, NULL_ARM_JOINT_POSITIONS)
    assert center.shape == (3,)
    assert center.min() > -1.0
    assert center.max() < 1.0


def test_workspace_bounds_center_matches_null_fk():
    sim = ArmSim()
    bounds = build_workspace_bounds(sim.model, sim.data, sim.controller)
    null_center = fk_ee_position(sim.model, sim.data, sim.controller, NULL_ARM_JOINT_POSITIONS)
    assert bounds.center.shape == (3,)
    assert (bounds.center == null_center).all()


def test_workspace_bounds_accept_explicit_numpy_half_extents():
    sim = ArmSim()
    extents = np.array([0.1, 0.2, 0.3])
    bounds = build_workspace_bounds(sim.model, sim.data, sim.controller, extents)
    assert np.allclose(bounds.half_extents, extents)
