#ifndef DYNAMIC_ACTION_ENVELOPE_HPP
#define DYNAMIC_ACTION_ENVELOPE_HPP

#include "policy_parameters.hpp"
#include "robot_parameters.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

class DynamicActionEnvelope {
 public:
  static constexpr double kControlDt = 0.02;
  static constexpr double kPositionMargin = 0.002;
  static constexpr double kDynamicMarginRatio = 0.05;

  DynamicActionEnvelope() { Reset(); }

  void Reset() {
    for (size_t i = 0; i < G1_NUM_MOTOR; ++i) previous_target_[i] = default_angles[i];
    previous_velocity_.fill(0.0);
  }

  std::array<double, G1_NUM_MOTOR> Transform(
      const std::array<double, G1_NUM_MOTOR>& raw_action) {
    std::array<double, G1_NUM_MOTOR> target{};
    const double acceleration = G1_MAX_TARGET_ACCELERATION * (1.0 - kDynamicMarginRatio);
    const double velocity_limit = G1_MAX_TARGET_VELOCITY * (1.0 - kDynamicMarginRatio);
    const double adt = acceleration * kControlDt;

    for (size_t i = 0; i < G1_NUM_MOTOR; ++i) {
      if (!std::isfinite(raw_action[i])) throw std::invalid_argument("non-finite policy action");
      const double low = G1_JOINT_LOWER[i] + kPositionMargin;
      const double high = G1_JOINT_UPPER[i] - kPositionMargin;
      const double offset = default_angles[i];
      const double scaled = raw_action[i] * g1_action_scale[i];
      const double desired = offset + (raw_action[i] >= 0.0
          ? (high - offset) * std::tanh(scaled / (high - offset))
          : (offset - low) * std::tanh(scaled / (offset - low)));

      const double upper_distance = std::max(0.0, high - previous_target_[i]);
      const double lower_distance = std::max(0.0, previous_target_[i] - low);
      const double max_upper_speed = -adt + std::sqrt(adt * adt + 2.0 * acceleration * upper_distance);
      const double max_lower_speed = -adt + std::sqrt(adt * adt + 2.0 * acceleration * lower_distance);
      const double velocity_low = std::max({previous_velocity_[i] - adt, -velocity_limit, -max_lower_speed});
      const double velocity_high = std::min({previous_velocity_[i] + adt, velocity_limit, max_upper_speed});
      if (velocity_low > velocity_high) throw std::runtime_error("no dynamically feasible successor");

      const double desired_velocity = (desired - previous_target_[i]) / kControlDt;
      const double midpoint = 0.5 * (velocity_low + velocity_high);
      const double half_range = 0.5 * (velocity_high - velocity_low);
      const double next_velocity = half_range > 0.0
          ? midpoint + half_range * std::tanh((desired_velocity - midpoint) / half_range)
          : midpoint;
      target[i] = previous_target_[i] + next_velocity * kControlDt;
      previous_velocity_[i] = next_velocity;
    }
    previous_target_ = target;
    return target;
  }

  const std::array<double, G1_NUM_MOTOR>& previous_velocity() const {
    return previous_velocity_;
  }

 private:
  std::array<double, G1_NUM_MOTOR> previous_target_{};
  std::array<double, G1_NUM_MOTOR> previous_velocity_{};
};

#endif
