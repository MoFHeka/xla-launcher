/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#include "xla_launcher/runtime/pjrt_computation_client.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla_launcher/runtime/computation_client.hpp"
#include "xla_launcher/runtime/tensor_source.hpp"

namespace xla_launcher {
namespace runtime {

class PjRtComputationClientTest : public ::testing::TestWithParam<std::string> {
 protected:
  void SetUp() override {
    std::string device_type = GetParam();
    setenv("PJRT_DEVICE", device_type.c_str(), 1);
    client_ = std::make_unique<PjRtComputationClient>(ClientOptions{});
    auto pjrt_client = client_->client_.get();
    std::cout << "client_->platform_name(): " << pjrt_client->platform_name()
              << std::endl;
    std::cout << "Device count: " << pjrt_client->device_count() << std::endl;
    for (auto* device : pjrt_client->devices()) {
      std::cout << "Device: " << device->DebugString() << std::endl;
    }
    auto devices_vec = client_->GetLocalDevices();
    bool has_cuda = false;
    for (size_t i = 0; i < devices_vec.size(); i++) {
      std::cout << "devices_[" << i << "]: " << devices_vec[i] << std::endl;
      devices_[device_type] = devices_vec[i];
      if (devices_vec[i].find("cuda") != std::string::npos) {
        has_cuda = true;
      }
    }
    if (device_type == "cuda" && !has_cuda) {
      GTEST_SKIP() << "No CUDA device available, skipping CUDA test.";
    }
  }

  std::unique_ptr<PjRtComputationClient> client_;
  std::map<std::string, std::string> devices_;
};

INSTANTIATE_TEST_SUITE_P(
  DeviceTypes, PjRtComputationClientTest, ::testing::Values("cpu", "cuda"));

// Returns a computation to compute x + y where x and y are both F32[2,2]
// arrays.
absl::StatusOr<xla::XlaComputation> MakeAddComputation() {
  const xla::Shape input_shape =
    xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {2, 2});
  xla::XlaBuilder builder("AddComputation");
  xla::XlaOp x = xla::Parameter(&builder, 0, input_shape, "x");
  xla::XlaOp y = xla::Parameter(&builder, 1, input_shape, "y");
  xla::XlaOp sum = xla::Add(x, y);
  return builder.Build();
}

absl::StatusOr<xla::XlaComputation> MakeInvalidComputation() {
  const xla::Shape shape =
    xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {2, 2});
  xla::XlaBuilder builder("InvalidCustomCallComputation");
  xla::XlaOp x = xla::Parameter(&builder, 0, shape, "x");
  xla::XlaOp invalid =
    xla::CustomCall(&builder, "non_existent_custom_call", {x}, shape);
  return builder.Build();
}

TEST_P(PjRtComputationClientTest, ThrowsExpectedExceptionWhenCompileFails) {
  std::string device_type = GetParam();
  // Compose a computation to add two matrices.
  xla::Shape out_shape(
    xla::F32, {2, 2},
    /*dynamic_dimensions=*/{});
  std::vector<CompileInstance> instances;
  instances.push_back(CompileInstance(
    std::move(MakeInvalidComputation().value()), devices_[device_type],
    client_->GetCompilationDevices(
      devices_[device_type], client_->GetLocalDevices()),
    &out_shape));

  // Compiling the graph should fail, which should throw instead of crashing.
  EXPECT_THROW(client_->Compile(std::move(instances)), std::invalid_argument);
}

TEST_P(PjRtComputationClientTest, AddComputationWorks) {
  std::string device_type = GetParam();
  // Compose a computation to add two 2x2 matrices.
  auto out_shape = xla::ShapeUtil::MakeShape(xla::F32, {2, 2});
  std::vector<CompileInstance> instances;
  instances.push_back(CompileInstance(
    std::move(MakeAddComputation().value()), devices_[device_type],
    client_->GetCompilationDevices(
      devices_[device_type], client_->GetLocalDevices()),
    &out_shape));

  // Prepare inputs.
  xla::Literal literal_x =
    xla::LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}});
  xla::Literal literal_y =
    xla::LiteralUtil::CreateR2<float>({{5.0f, 6.0f}, {7.0f, 8.0f}});

  // Compile the graph.
  std::vector<ComputationClient::ComputationPtr> computations =
    client_->Compile(std::move(instances));

  // Copy inputs to device.
  ComputationClient::ExecuteComputationOptions options{};
  std::vector<std::shared_ptr<const TensorSource>> args = {
    std::make_shared<LiteralSource>(
      std::move(literal_x), devices_[device_type]),
    std::make_shared<LiteralSource>(
      std::move(literal_y), devices_[device_type])};

  // Execute the graph.
  std::vector<ComputationClient::DataPtr> results = client_->ExecuteComputation(
    *computations[0], client_->TransferToDevice(absl::MakeConstSpan(args)),
    devices_[device_type], options);

  // Copy the output from device back to host and assert correctness.
  ASSERT_EQ(results.size(), 1);
  auto result_literals = client_->TransferFromDevice(results);
  ASSERT_THAT(result_literals, ::testing::SizeIs(1));
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(
    xla::LiteralUtil::CreateR2<float>({{6.0f, 8.0f}, {10.0f, 12.0f}}),
    result_literals[0]));
}

}  // namespace runtime
}  // namespace xla_launcher

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
