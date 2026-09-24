#include <gtest/gtest.h>

#include "motor_safety.hpp"

#include <limits>
#include <set>

TEST(MotorSafety, NamedMapIsCompleteAndUnique) {
  EXPECT_EQ(G1_MOTOR_NAMES.size(), G1_NUM_MOTOR);
  std::set<std::string_view> names(G1_MOTOR_NAMES.begin(), G1_MOTOR_NAMES.end());
  EXPECT_EQ(names.size(), G1_NUM_MOTOR);
}

TEST(MotorSafety, RejectsInvalidAndStaleTargetsAndLimitsDerivatives) {
  MotorSafety safety;
  MotorCommand command;
  const auto t0 = MotorSafety::Clock::now();
  std::string reason;
  EXPECT_TRUE(safety.ValidateAndLimit(command, t0, t0, reason));

  command.q_target[LeftAnkleRoll] = 0.4f;
  EXPECT_FALSE(safety.ValidateAndLimit(command, t0 + std::chrono::milliseconds(20),
                                       t0 + std::chrono::milliseconds(20), reason));
  EXPECT_EQ(reason, "left_ankle_roll_joint: position 0.4 outside [-0.2618, 0.2618]");

  command.q_target[LeftAnkleRoll] = 0.0f;
  command.q_target[LeftHipPitch] = 0.5f;
  EXPECT_TRUE(safety.ValidateAndLimit(command, t0 + std::chrono::milliseconds(20),
                                      t0 + std::chrono::milliseconds(20), reason));
  EXPECT_EQ(reason, "target derivatives limited");
  EXPECT_FLOAT_EQ(command.q_target[LeftHipPitch], 0.032f);

  safety.Reset();
  command.q_target[LeftHipPitch] = 0.0f;
  EXPECT_FALSE(safety.ValidateAndLimit(command, t0, t0 + std::chrono::milliseconds(101), reason));
  command.q_target[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(safety.ValidateAndLimit(command, t0, t0, reason));
}

TEST(MotorSafety, RejectsDerivativeLimitedOvershoot) {
  MotorSafety safety;
  MotorCommand command;
  std::string reason;
  const auto t0 = MotorSafety::Clock::now();
  EXPECT_TRUE(safety.ValidateAndLimit(command, t0, t0, reason));

  bool rejected = false;
  for (int step = 1; step <= 20; ++step) {
    command.q_target[LeftAnkleRoll] = G1_JOINT_LOWER[LeftAnkleRoll];
    const auto time = t0 + std::chrono::milliseconds(20 * step);
    if (!safety.ValidateAndLimit(command, time, time, reason)) {
      rejected = true;
      break;
    }
  }
  EXPECT_TRUE(rejected);
  EXPECT_EQ(reason,
            "left_ankle_roll_joint: derivative limits cannot preserve position limit");
}
