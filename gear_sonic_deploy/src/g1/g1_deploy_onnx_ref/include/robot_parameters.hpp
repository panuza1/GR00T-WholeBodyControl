/**
 * @file robot_parameters.hpp
 * @brief Hardware-level constants, data structures, and joint indices for the
 *        Unitree G1 humanoid robot.
 *
 * This header defines:
 *  - DDS topic names for the Unitree SDK (low-level command / state channels).
 *  - Motor count and per-joint MotorCommand structure.
 *  - HeadingState – a compact struct bundling the captured IMU quaternion
 *    with a user-controlled delta-heading offset.
 *  - OperatorState – high-level start / stop / play flags.
 *  - Control mode enum (series vs. parallel ankle actuation).
 *  - G1JointIndex enum mapping symbolic joint names to hardware motor indices.
 *
 * All indices in this file use the **hardware / URDF ordering** (not the
 * IsaacLab training ordering – see `policy_parameters.hpp` for the mapping).
 */

#ifndef ROBOT_PARAMETERS_HPP
#define ROBOT_PARAMETERS_HPP

#include <array>
#include <string>
#include <string_view>

// ---------------------------------------------------------------------------
// Unitree SDK DDS topic names
// ---------------------------------------------------------------------------
static const std::string HG_CMD_TOPIC = "rt/lowcmd";       ///< Low-level motor command topic.
static const std::string HG_IMU_TORSO = "rt/secondary_imu";///< Secondary (torso) IMU topic.
static const std::string HG_STATE_TOPIC = "rt/lowstate";    ///< Low-level motor / sensor state topic.

/// Total number of actuated joints on the G1 (29-DOF configuration).
const int G1_NUM_MOTOR = 29;

inline constexpr std::array<std::string_view, G1_NUM_MOTOR> G1_MOTOR_NAMES = {
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint"};

// Authoritative G1 29-DOF ranges from g1_29dof_with_hand.xml, in hardware order.
// All controller safety checks intentionally share these arrays.
inline constexpr std::array<float, G1_NUM_MOTOR> G1_JOINT_LOWER = {
    -2.5307f,-0.5236f,-2.7576f,-0.087267f,-0.87267f,-0.2618f,
    -2.5307f,-2.9671f,-2.7576f,-0.087267f,-0.87267f,-0.2618f,
    -2.618f,-0.52f,-0.52f,-3.0892f,-1.5882f,-2.618f,-1.0472f,
    -1.97222f,-1.61443f,-1.61443f,-3.0892f,-2.2515f,-2.618f,
    -1.0472f,-1.97222f,-1.61443f,-1.61443f};
inline constexpr std::array<float, G1_NUM_MOTOR> G1_JOINT_UPPER = {
    2.8798f,2.9671f,2.7576f,2.8798f,0.5236f,0.2618f,
    2.8798f,0.5236f,2.7576f,2.8798f,0.5236f,0.2618f,
    2.618f,0.52f,0.52f,2.6704f,2.2515f,2.618f,2.0944f,
    1.97222f,1.61443f,1.61443f,2.6704f,1.5882f,2.618f,
    2.0944f,1.97222f,1.61443f,1.61443f};

inline constexpr float G1_MAX_TARGET_VELOCITY = 8.0f;
inline constexpr float G1_MAX_TARGET_ACCELERATION = 80.0f;

/**
 * @brief Per-joint motor command sent to the low-level controller.
 *
 * Each field is an array of size G1_NUM_MOTOR (29), indexed by hardware joint
 * index (see G1JointIndex).
 */
struct MotorCommand {
    std::array<float, G1_NUM_MOTOR> q_target = {};   ///< Target position (rad).
    std::array<float, G1_NUM_MOTOR> dq_target = {};  ///< Target velocity (rad/s).
    std::array<float, G1_NUM_MOTOR> kp = {};          ///< Position gain (Nm/rad).
    std::array<float, G1_NUM_MOTOR> kd = {};          ///< Velocity gain (Nm·s/rad).
    std::array<float, G1_NUM_MOTOR> tau_ff = {};      ///< Feed-forward torque (Nm).
};

/**
 * @brief Bundled heading state for thread-safe access via DataBuffer.
 *
 * Captures both the initial IMU base quaternion (set when heading is
 * reinitialised) and a user-adjustable delta heading offset (adjusted via
 * keyboard Q/E or D-pad).
 */
struct HeadingState {
    std::array<double, 4> init_base_quat;  ///< Captured IMU base quaternion (w,x,y,z) at init.
    double delta_heading;                   ///< Cumulative heading offset (radians).
    
    HeadingState(const std::array<double, 4>& quat = {1.0, 0.0, 0.0, 0.0}, double delta = 0.0)
        : init_base_quat(quat), delta_heading(delta) {}
};

/**
 * @brief High-level operator signals (set by input interfaces, read by control loop).
 */
struct OperatorState {
  bool stop = false;   ///< Emergency stop requested.
  bool start = false;  ///< Control-system start requested.
  bool play = false;   ///< Motion playback active.
};

/**
 * @brief Ankle actuation mode.
 *
 * The G1's ankle uses a coupled 2-DOF mechanism that can be controlled
 * in series (pitch/roll) or parallel (A/B motor) modes.
 */
enum class Mode {
  PR = 0, ///< Series control for Pitch / Roll joints.
  AB = 1  ///< Parallel control for A / B motors.
};

/**
 * @brief Symbolic names for G1 hardware joint indices (0–28).
 *
 * Ankle joints have dual names because they can be addressed in either
 * series (Pitch/Roll) or parallel (A/B) mode.  Joints marked "INVALID"
 * are not present on the 23-DOF or waist-locked 29-DOF variants.
 */
enum G1JointIndex {
  LeftHipPitch = 0,
  LeftHipRoll = 1,
  LeftHipYaw = 2,
  LeftKnee = 3,
  LeftAnklePitch = 4,
  LeftAnkleB = 4,
  LeftAnkleRoll = 5,
  LeftAnkleA = 5,
  RightHipPitch = 6,
  RightHipRoll = 7,
  RightHipYaw = 8,
  RightKnee = 9,
  RightAnklePitch = 10,
  RightAnkleB = 10,
  RightAnkleRoll = 11,
  RightAnkleA = 11,
  WaistYaw = 12,
  WaistRoll = 13, // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistA = 13, // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistPitch = 14, // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistB = 14, // NOTE INVALID for g1 23dof/29dof with waist locked
  LeftShoulderPitch = 15,
  LeftShoulderRoll = 16,
  LeftShoulderYaw = 17,
  LeftElbow = 18,
  LeftWristRoll = 19,
  LeftWristPitch = 20, // NOTE INVALID for g1 23dof
  LeftWristYaw = 21, // NOTE INVALID for g1 23dof
  RightShoulderPitch = 22,
  RightShoulderRoll = 23,
  RightShoulderYaw = 24,
  RightElbow = 25,
  RightWristRoll = 26,
  RightWristPitch = 27, // NOTE INVALID for g1 23dof
  RightWristYaw = 28 // NOTE INVALID for g1 23dof
};

#endif // ROBOT_PARAMETERS_HPP
