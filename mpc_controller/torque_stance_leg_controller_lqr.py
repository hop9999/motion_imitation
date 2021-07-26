# Lint as: python3
"""A torque based stance controller framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from motion_imitation.robots.a1 import foot_position_in_hip_frame

from typing import Any, Sequence, Tuple
import time

import numpy as np
# import time
from casadi import *
import scipy.linalg as la
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import leg_controller
from mpc_controller import qp_torque_optimizer

_FORCE_DIMENSION = 3
KP = np.array((0., 0., 100., 100., 100., 0.))
KD = np.array((40., 30., 10., 10., 10., 30.))
MAX_DDQ = np.array((10., 10., 10., 20., 20., 20.))
MIN_DDQ = -MAX_DDQ
MPC_BODY_MASS = 108 / 9.8
MPC_BODY_INERTIA = np.array((0.017, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 4.
  


class TorqueStanceLegController(leg_controller.LegController):
  """A torque based stance leg controller framework.
  Takes in high level parameters like walking speed and turning speed, and
  generates necessary the torques for stance legs.
  """
  def __init__(
      self,
      robot: Any,
      gait_generator: Any,
      state_estimator: Any,
      num_legs: int = 4,
      friction_coeffs: Sequence[float] = (0.45, 0.45, 0.45, 0.45),
  ):
    """Initializes the class.
    Tracks the desired position/velocity of the robot by computing proper joint
    torques using MPC module.
    Args:
      robot: A robot instance.
      gait_generator: Used to query the locomotion phase and leg states.
      state_estimator: Estimate the robot states (e.g. CoM velocity).
      desired_speed: desired CoM speed in x-y plane.
      desired_twisting_speed: desired CoM rotating speed in z direction.
      desired_body_height: The standing height of the robot.
      body_mass: The total mass of the robot.
      body_inertia: The inertia matrix in the body principle frame. We assume
        the body principle coordinate frame has x-forward and z-up.
      num_legs: The number of legs used for force planning.
      friction_coeffs: The friction coeffs on the contact surfaces.
    """
    self._robot = robot
    self._gait_generator = gait_generator
    self._state_estimator = state_estimator
    self.desired_q = np.zeros((12,1))
    self._num_legs = num_legs
    self._friction_coeffs = np.array(friction_coeffs)
    self.K_dain = np.array((12,12))
    self.force_des = np.array((4,3))

  def reset(self, current_time):
    del current_time

  def update(self, current_time):
    del current_time

  def _estimate_robot_height(self, contacts):
    if np.sum(contacts) == 0:
      # All foot in air, no way to estimate
      return self._desired_body_height
    else:
      base_orientation = self._robot.GetBaseOrientation()
      rot_mat = self._robot.pybullet_client.getMatrixFromQuaternion(
          base_orientation)
      rot_mat = np.array(rot_mat).reshape((3, 3))

      foot_positions = self._robot.GetFootPositionsInBaseFrame()
      foot_positions_world_frame = (rot_mat.dot(foot_positions.T)).T
      # pylint: disable=unsubscriptable-object
      useful_heights = contacts * (-foot_positions_world_frame[:, 2])
      return np.sum(useful_heights) / np.sum(contacts)

  def get_action(self):
    """Computes the torque for stance legs."""
    # Actual q and dq
    contacts = np.array(
        [(leg_state in (gait_generator_lib.LegState.STANCE,
                        gait_generator_lib.LegState.EARLY_CONTACT))
         for leg_state in self._gait_generator.desired_leg_state],
        dtype=np.int32)
    
    #robot_com_position = np.array(self._robot.GetBasePosition())#np.array(
    #    (0., 0., self._estimate_robot_height(contacts)))
    p = self._robot.GetFootPositionsInBaseFrame()
    robot_com_position = np.array(
      (-np.mean(p[:,0]), -np.mean(p[:,1]), self._estimate_robot_height(contacts)))
    
    robot_com_velocity = self._state_estimator.com_velocity_body_frame
    robot_com_roll_pitch_yaw = np.array(self._robot.GetBaseRollPitchYaw())
    robot_com_roll_pitch_yaw[2] = 0  # To prevent yaw drifting
    robot_com_roll_pitch_yaw_rate = self._robot.GetBaseRollPitchYawRate()
    robot_q = np.hstack((robot_com_position, robot_com_roll_pitch_yaw))
    robot_dq = np.hstack((robot_com_velocity, robot_com_roll_pitch_yaw_rate))
    
    X = np.hstack((robot_q,robot_dq))
    X_des = self.desired_q[:,0]
    
    e = X_des - X
    print(e)
    if np.max(np.abs(e[0:3])) > 0.03 or np.max(np.abs(e[3:6])) > 0.1 or np.max(np.abs(e[6:12])) > 1.5:
      raise RuntimeError('Convergence error')

    f_lin = self.K_gain @ e
    f_fb = f_lin.reshape((4,3))

    contact_forces = self.force_des + f_fb
    #print(contact_forces)

    action = {}
    for leg_id, force in enumerate(contact_forces):
      # While "Lose Contact" is useful in simulation, in real environment it's
      # susceptible to sensor noise. Disabling for now.
      # if self._gait_generator.leg_state[
      #     leg_id] == gait_generator_lib.LegState.LOSE_CONTACT:
      #   force = (0, 0, 0)
      motor_torques = self._robot.MapContactForceToJointTorques(leg_id, force)
      for joint_id, torque in motor_torques.items():
        action[joint_id] = (0, 0, 0, 0, torque)
    
    return action, contact_forces

def prepare_point(contacts, foot_position, des_acceleration):
  dt = 0.015#robot.time_step
  mass_matrix = qp_torque_optimizer.compute_mass_matrix(
                    MPC_BODY_MASS,
                    np.array(MPC_BODY_INERTIA).reshape((3, 3)),
                    foot_position)
  A_f = np.vstack((np.hstack((np.zeros((6,6)),np.eye((6)))),np.zeros((6,12))))
  B_f = np.vstack((np.zeros((6, 12)),mass_matrix))
  A = A_f*dt + np.eye(12)
  B = B_f*dt
  Q = 1*np.diag([1e6,1e6,1e6,  1e6,1e6,1e6,  1e3,1e3,1e3,  1e3,1e3,1e3])
  R = 0.1*np.eye(12)
  P = la.solve_discrete_are(A, B, Q, R)
  K_gain = -np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
  force_des = qp_torque_optimizer.compute_contact_force(
                MPC_BODY_MASS, MPC_BODY_INERTIA, des_acceleration, contacts, foot_position) 
  return K_gain, force_des