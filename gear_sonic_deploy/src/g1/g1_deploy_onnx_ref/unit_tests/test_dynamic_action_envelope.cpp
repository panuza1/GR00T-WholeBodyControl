#include <gtest/gtest.h>

#include "dynamic_action_envelope.hpp"

#include <array>
#include <cmath>

TEST(DynamicActionEnvelope, IsPositionVelocityAndAccelerationFeasible) {
  DynamicActionEnvelope envelope;
  std::array<double, G1_NUM_MOTOR> raw{};
  std::array<double, G1_NUM_MOTOR> previous_target = default_angles;
  std::array<double, G1_NUM_MOTOR> previous_velocity{};

  for (int step = 0; step < 1000; ++step) {
    for (size_t i = 0; i < raw.size(); ++i) raw[i] = ((step + i) % 2 == 0) ? 100.0 : -100.0;
    const auto target = envelope.Transform(raw);
    const auto& velocity = envelope.previous_velocity();
    for (size_t i = 0; i < target.size(); ++i) {
      EXPECT_GT(target[i], G1_JOINT_LOWER[i]);
      EXPECT_LT(target[i], G1_JOINT_UPPER[i]);
      EXPECT_LE(std::abs(velocity[i]), G1_MAX_TARGET_VELOCITY * 0.95 + 1e-9);
      EXPECT_LE(std::abs(velocity[i] - previous_velocity[i]) / 0.02,
                G1_MAX_TARGET_ACCELERATION * 0.95 + 1e-8);
    }
    previous_target = target;
    previous_velocity = velocity;
  }
}

TEST(DynamicActionEnvelope, UsesAsymmetricRangesAndPreservesZeroPose) {
  DynamicActionEnvelope envelope;
  std::array<double, G1_NUM_MOTOR> raw{};
  auto target = envelope.Transform(raw);
  for (size_t i = 0; i < target.size(); ++i) EXPECT_NEAR(target[i], default_angles[i], 1e-12);

  raw[LeftAnklePitch] = 100.0;
  for (int i = 0; i < 200; ++i) target = envelope.Transform(raw);
  EXPECT_LT(target[LeftAnklePitch], G1_JOINT_UPPER[LeftAnklePitch]);
  EXPECT_GT(target[LeftAnklePitch], 0.0);

  raw[LeftAnklePitch] = -100.0;
  for (int i = 0; i < 400; ++i) target = envelope.Transform(raw);
  EXPECT_GT(target[LeftAnklePitch], G1_JOINT_LOWER[LeftAnklePitch]);
  EXPECT_LT(target[LeftAnklePitch], -0.5);
}
