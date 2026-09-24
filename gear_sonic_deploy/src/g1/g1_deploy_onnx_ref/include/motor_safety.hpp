#ifndef MOTOR_SAFETY_HPP
#define MOTOR_SAFETY_HPP

#include "robot_parameters.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <sstream>
#include <string>

class MotorSafety {
 public:
  using Clock = std::chrono::steady_clock;

  bool ValidateAndLimit(MotorCommand& command, Clock::time_point command_time,
                        Clock::time_point now, std::string& reason) {
    if (command_time == Clock::time_point{} || now < command_time ||
        now - command_time > std::chrono::milliseconds(100)) {
      reason = "stale motor command";
      return false;
    }
    for (size_t i = 0; i < G1_NUM_MOTOR; ++i) {
      if (!std::isfinite(command.q_target[i]) || !std::isfinite(command.dq_target[i]) ||
          !std::isfinite(command.kp[i]) || !std::isfinite(command.kd[i]) ||
          !std::isfinite(command.tau_ff[i])) {
        reason = std::string(G1_MOTOR_NAMES[i]) + ": non-finite command";
        return false;
      }
      if (command.q_target[i] < G1_JOINT_LOWER[i] || command.q_target[i] > G1_JOINT_UPPER[i]) {
        std::ostringstream message;
        message << G1_MOTOR_NAMES[i] << ": position " << command.q_target[i]
                << " outside [" << G1_JOINT_LOWER[i] << ", " << G1_JOINT_UPPER[i] << "]";
        reason = message.str();
        return false;
      }
      if (std::fabs(command.dq_target[i]) > G1_MAX_TARGET_VELOCITY) {
        reason = std::string(G1_MOTOR_NAMES[i]) + ": velocity limit";
        return false;
      }
    }
    bool limited = false;
    if (have_previous_ && command_time != previous_time_) {
      const float dt = std::chrono::duration<float>(command_time - previous_time_).count();
      if (!(dt > 0.0f && dt <= 0.2f)) {
        reason = "invalid command interval";
        return false;
      }
      auto next_q = command.q_target;
      auto next_velocity = previous_velocity_;
      for (size_t i = 0; i < G1_NUM_MOTOR; ++i) {
        const float requested_velocity = (command.q_target[i] - previous_q_[i]) / dt;
        const float velocity = std::clamp(requested_velocity,
            std::max(-G1_MAX_TARGET_VELOCITY,
                     previous_velocity_[i] - G1_MAX_TARGET_ACCELERATION * dt),
            std::min(G1_MAX_TARGET_VELOCITY,
                     previous_velocity_[i] + G1_MAX_TARGET_ACCELERATION * dt));
        limited |= velocity != requested_velocity;
        next_q[i] = previous_q_[i] + velocity * dt;
        next_velocity[i] = velocity;
        if (next_q[i] < G1_JOINT_LOWER[i] || next_q[i] > G1_JOINT_UPPER[i]) {
          reason = std::string(G1_MOTOR_NAMES[i]) +
                   ": derivative limits cannot preserve position limit";
          return false;
        }
      }
      command.q_target = next_q;
      previous_velocity_ = next_velocity;
    }
    previous_q_ = command.q_target;
    previous_time_ = command_time;
    have_previous_ = true;
    reason = limited ? "target derivatives limited" : "";
    return true;
  }

  void Reset() {
    previous_q_.fill(0.0f);
    previous_velocity_.fill(0.0f);
    previous_time_ = {};
    have_previous_ = false;
  }

 private:
  std::array<float, G1_NUM_MOTOR> previous_q_{};
  std::array<float, G1_NUM_MOTOR> previous_velocity_{};
  Clock::time_point previous_time_{};
  bool have_previous_ = false;
};

#endif
