#include "test_support.hpp"

#include "flowguard/risk_control.hpp"

using namespace flowguard;
using flowguard::test::require;

TEST_CASE("clear centre continues forward") {
  AvoidanceController controller;
  RiskAssessment clear;
  const auto command = controller.update(clear, 1.0 / 30.0);
  require(command.speed_mps > 1.9F, "clear path should preserve target speed");
  require(command.brake < 0.1F, "clear path should not brake");
}

TEST_CASE("blocked centre steers toward lower-risk side") {
  AvoidanceController controller;
  RiskAssessment risk;
  risk.sectors = {0.1F, 0.8F, 0.7F};
  risk.warning = WarningLevel::Yellow;
  controller.update(risk, 1.0 / 30.0);
  controller.update(risk, 1.0 / 30.0);
  const auto command = controller.update(risk, 1.0 / 30.0);
  require(command.yaw_rate_deg_s > 0.0F, "controller should turn left when left sector is safer");
}

TEST_CASE("symmetric centre threat uses deterministic right turn") {
  AvoidanceController controller;
  RiskAssessment risk;
  risk.sectors = {0.0F, 0.8F, 0.0F};
  risk.warning = WarningLevel::Red;
  controller.update(risk, 1.0 / 30.0);
  controller.update(risk, 1.0 / 30.0);
  const auto command = controller.update(risk, 1.0 / 30.0);
  require(command.yaw_rate_deg_s < 0.0F, "symmetric threat tie-break changed");
}

TEST_CASE("critical side motion does not block a clear centre") {
  AvoidanceController controller;
  RiskAssessment risk;
  risk.sectors = {0.9F, 0.0F, 0.0F};
  risk.warning = WarningLevel::Red;
  const auto command = controller.update(risk, 1.0 / 30.0);
  require(command.brake < 0.1F, "side-only motion caused critical braking");
  require(command.speed_mps > 1.9F, "side-only motion reduced forward speed");
}

TEST_CASE("controller cannot receive evaluation ground truth") {
  AvoidanceController controller;
  RiskAssessment risk;
  risk.sectors = {0.9F, 0.9F, 0.9F};
  risk.warning = WarningLevel::Red;
  controller.update(risk, 1.0 / 30.0);
  controller.update(risk, 1.0 / 30.0);
  const auto command = controller.update(risk, 1.0 / 30.0);
  require(command.brake > 0.0F, "blocked route should begin braking");
}
