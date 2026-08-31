"""Standard Menagerie Panda tendon driver; deliberately rejects the MJX variant."""

from typing import Optional

import mujoco
import numpy as np

from mimic.common.types import GripperFeedback


def descendants(model: mujoco.MjModel, body_id: int) -> set[int]:
    result = {body_id}
    for i in range(body_id + 1, model.nbody):
        if int(model.body_parentid[i]) in result:
            result.add(i)
    return result


class PandaGripperDriver:
    open_width_m = 0.08
    closed_width_m = 0.0

    def __init__(
        self,
        model: mujoco.MjModel,
        actuator_name: str = "actuator8",
        finger_joint_names: tuple[str, str] = ("finger_joint1", "finger_joint2"),
    ):
        self.model = model
        self.actuator_names = (actuator_name,)
        a = model.actuator(actuator_name).id
        self.joint_ids = [model.joint(name).id for name in finger_joint_names]
        self.qpos_ids = model.jnt_qposadr[self.joint_ids]
        self.dof_ids = model.jnt_dofadr[self.joint_ids]
        self.finger_bodies = [descendants(model, int(model.jnt_bodyid[j])) for j in self.joint_ids]
        if len(set(self.joint_ids)) != 2:
            raise ValueError("Panda driver needs two distinct finger joints")
        if (
            not np.all(model.jnt_type[self.joint_ids] == mujoco.mjtJoint.mjJNT_SLIDE)
            or not np.all(model.jnt_limited[self.joint_ids])
            or not np.allclose(
                model.jnt_range[self.joint_ids], [[0, 0.04], [0, 0.04]], rtol=0, atol=1e-12
            )
        ):
            raise ValueError("Unexpected Panda finger ranges/types")
        if (
            model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_TENDON
            or not model.actuator_ctrllimited[a]
            or not np.array_equal(model.actuator_ctrlrange[a], [0, 255])
            or not np.array_equal(model.actuator_gear[a], [1, 0, 0, 0, 0, 0])
            or model.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE
            or model.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED
            or model.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE
            or not np.allclose(
                model.actuator_gainprm[a, :3], [0.01568627451, 0, 0], rtol=0, atol=1e-12
            )
            or not np.array_equal(model.actuator_biasprm[a, :3], [0, -100, -10])
            or not model.actuator_forcelimited[a]
            or not np.array_equal(model.actuator_forcerange[a], [-100, 100])
        ):
            raise ValueError("Panda gripper mapping mismatch; standard tendon model required")
        tendon = model.actuator_trnid[a, 0]
        adr, count = model.tendon_adr[tendon], model.tendon_num[tendon]
        wraps = slice(adr, adr + count)
        if (
            count != 2
            or set(model.wrap_objid[wraps]) != set(self.joint_ids)
            or not np.all(model.wrap_type[wraps] == mujoco.mjtWrap.mjWRAP_JOINT)
            or not np.array_equal(model.wrap_prm[wraps], [0.5, 0.5])
        ):
            raise ValueError("Expected equal finger coupling through the Panda tendon")
        coupled = any(
            model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
            and model.eq_active0[i]
            and {model.eq_obj1id[i], model.eq_obj2id[i]} == set(self.joint_ids)
            and np.array_equal(model.eq_data[i, :5], [0, 1, 0, 0, 0])
            for i in range(model.neq)
        )
        if not coupled:
            raise ValueError("Expected active Panda finger equality constraint")

    def controls(self, width_m: float) -> dict[str, float]:
        if not np.isfinite(width_m) or not self.closed_width_m <= width_m <= self.open_width_m:
            raise ValueError("Panda nominal width must be in [0, 0.08] meters; not clipped")
        return {self.actuator_names[0]: 255.0 * (width_m / self.open_width_m)}

    def observe(self, data: mujoco.MjData, object_body_id: Optional[int]) -> GripperFeedback:
        forces = np.zeros(2)
        if object_body_id is not None:
            object_bodies = descendants(self.model, object_body_id)
            force = np.empty(6)
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                bodies = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
                if contact.efc_address < 0:
                    continue
                for finger, group in enumerate(self.finger_bodies):
                    if (bodies[0] in group and bodies[1] in object_bodies) or (
                        bodies[1] in group and bodies[0] in object_bodies
                    ):
                        mujoco.mj_contactForce(self.model, data, contact_id, force)
                        forces[finger] += max(0.0, float(force[0]))
        return GripperFeedback(
            float(sum(data.qpos[self.qpos_ids])), float(sum(data.qvel[self.dof_ids])), tuple(forces)
        )
