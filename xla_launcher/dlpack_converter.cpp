/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#include "xla_launcher/dlpack_converter.hpp"

#include <memory>
#include <utility>
#include <vector>

#include "xla_launcher/runtime/pjrt_computation_client.hpp"
#include "xla_launcher/runtime/runtime.hpp"

namespace xla_launcher {

struct DLPackTensor {
  ~DLPackTensor();
  std::unique_ptr<xla::PjRtBuffer::ExternalReference> external_reference;
  std::shared_ptr<xla::PjRtBuffer> buffer_reference;

  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DLManagedTensor tensor;
};

DLPackTensor::~DLPackTensor() {
  if (external_reference) {
    external_reference.reset(nullptr);
  }
}

void DLPackTensorDeleter(DLManagedTensor* t) {
  if (t) {
    delete static_cast<DLPackTensor*>(t->manager_ctx);
  }
}

DLDeviceType DLDeviceTypeForDevice(const xla::PjRtDevice& device) {
  if (device.client()->platform_id() == xla::CpuId()) {
    return DLDeviceType::kDLCPU;
  } else if (device.client()->platform_id() == xla::CudaId()) {
    return DLDeviceType::kDLCUDA;
  }
  XLA_ERROR() << "Device " << device.DebugString()
              << " cannot be used as a DLPack device.";
}

// Reference: https://github.com/openxla/xla/blob/main/xla/python/dlpack.cc
DLDevice DLDeviceForDevice(const xla::PjRtDevice& device) {
  DLDevice dlDevice;
  dlDevice.device_type = DLDeviceTypeForDevice(device);
  dlDevice.device_id = device.local_hardware_id().value();
  return dlDevice;
}

// Reference: https://github.com/openxla/xla/blob/main/xla/python/dlpack.cc
std::optional<DLDataType> PrimitiveTypeToDLDataType(xla::PrimitiveType type) {
  switch (type) {
    case xla::PrimitiveType::S8:
      return DLDataType{kDLInt, 8, 1};
    case xla::PrimitiveType::S16:
      return DLDataType{kDLInt, 16, 1};
    case xla::PrimitiveType::S32:
      return DLDataType{kDLInt, 32, 1};
    case xla::PrimitiveType::S64:
      return DLDataType{kDLInt, 64, 1};
    case xla::PrimitiveType::U8:
      return DLDataType{kDLUInt, 8, 1};
    case xla::PrimitiveType::U16:
      return DLDataType{kDLUInt, 16, 1};
    case xla::PrimitiveType::U32:
      return DLDataType{kDLUInt, 32, 1};
    case xla::PrimitiveType::U64:
      return DLDataType{kDLUInt, 64, 1};
    case xla::PrimitiveType::F16:
      return DLDataType{kDLFloat, 16, 1};
    case xla::PrimitiveType::F32:
      return DLDataType{kDLFloat, 32, 1};
    case xla::PrimitiveType::F64:
      return DLDataType{kDLFloat, 64, 1};
    case xla::PrimitiveType::BF16:
      return DLDataType{kDLBfloat, 16, 1};
    case xla::PrimitiveType::PRED:
      return DLDataType{kDLBool, 8, 1};
    case xla::PrimitiveType::C64:
      return DLDataType{kDLComplex, 64, 1};
    case xla::PrimitiveType::C128:
      return DLDataType{kDLComplex, 128, 1};
    default:
      XLA_ERROR() << "XLA type " << xla::PrimitiveType_Name(type)
                  << " has no DLPack equivalent";
      return std::nullopt;
  }
}

std::vector<int64_t> StridesForShape(
  xla::PrimitiveType element_type, absl::Span<const int64_t> dimensions,
  const xla::Layout& layout) {
  XLA_CHECK_EQ(dimensions.size(), layout.minor_to_major().size());
  std::vector<int64_t> strides;
  strides.resize(dimensions.size());
  int64_t stride = 1;
  for (int i : layout.minor_to_major()) {
    strides[i] = stride;
    stride *= dimensions[i];
  }
  return strides;
}

// Convert an XLA tensor to a dlPack tensor.
DLManagedTensor* ToDLPack(const runtime::ComputationClient::DataPtr& input) {
  auto computation_client = runtime::GetComputationClient();
  XLA_CHECK_OK(computation_client.status())
    << "Failed to get computation client.";
  std::shared_ptr<xla::PjRtBuffer> pjrt_buffer =
    computation_client.value()->GetPjRtBuffer(input);
  XLA_CHECK(pjrt_buffer != nullptr) << "Could not get a valid pjrt_buffer";

  XLA_CHECK(!pjrt_buffer->IsTuple())
    << "Unimplemented. BufferToDLPackManagedTensor is not "
       "implemented for tuple buffers.";
  XLA_CHECK(!pjrt_buffer->has_dynamic_dimensions())
    << "Unimplemented. DynamicShape is not implemented in DLPack.";

  auto pack = std::make_unique<DLPackTensor>();
  DLTensor& dt = pack->tensor.dl_tensor;
  {
    // AcquireExternalReference may block
    auto external_ref = pjrt_buffer->AcquireExternalReference();
    XLA_CHECK_OK(external_ref.status());
    pack->external_reference = std::move(external_ref.value());
    xla::PjRtFuture<> future = pjrt_buffer->GetReadyFuture();
    absl::Status status = future.Await();
    XLA_CHECK_OK(status);
  }
  pack->buffer_reference = pjrt_buffer;

  dt.data = pack->external_reference->OpaqueDeviceMemoryDataPointer();
  pack->tensor.manager_ctx = pack.get();
  pack->tensor.deleter = DLPackTensorDeleter;
  dt.device = DLDeviceForDevice(*pjrt_buffer->device());
  dt.device.device_id = pjrt_buffer->device()->local_hardware_id().value();
  dt.ndim = pjrt_buffer->dimensions().size();
  auto dl_dtype = PrimitiveTypeToDLDataType(pjrt_buffer->element_type());
  XLA_CHECK(dl_dtype.has_value())
    << "Failed to convert XLA type to DLPack type";
  dt.dtype = dl_dtype.value();

  pack->shape = std::vector<int64_t>(
    pjrt_buffer->dimensions().begin(), pjrt_buffer->dimensions().end());
  xla::Layout xla_layout = pjrt_buffer->layout()->xla_layout();
  pack->strides = StridesForShape(
    pjrt_buffer->element_type(), pjrt_buffer->dimensions(), xla_layout);
  dt.shape = reinterpret_cast<std::int64_t*>(pack->shape.data());
  dt.strides = reinterpret_cast<std::int64_t*>(pack->strides.data());
  dt.byte_offset = 0;

  return &(pack.release()->tensor);
}

// Reference: https://github.com/openxla/xla/blob/main/xla/python/dlpack.cc
absl::StatusOr<xla::PjRtDevice*> DeviceForDLDevice(const DLDevice& context) {
  auto computation_client = runtime::GetComputationClient();
  XLA_CHECK_OK(computation_client.status())
    << "Failed to get computation client.";
  switch (context.device_type) {
    case DLDeviceType::kDLCPU:
      XLA_CHECK_EQ(computation_client.value()->GetPlatformID(), xla::CpuId());
      return computation_client.value()->LookupAddressableDevice(
        context.device_id);
    case DLDeviceType::kDLCUDA:
      XLA_CHECK_EQ(computation_client.value()->GetPlatformID(), xla::CudaId());
      return computation_client.value()->LookupAddressableDevice(
        context.device_id);
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
        "Unknown/unsupported DLPack device type %d", context.device_type));
  }
}

absl::StatusOr<xla::PjRtDevice*> DeviceForDLDevice(
  const DLDevice& context,
  const runtime::ComputationClient* computation_client) {
  switch (context.device_type) {
    case DLDeviceType::kDLCPU:
      XLA_CHECK_EQ(computation_client->GetPlatformID(), xla::CpuId());
      return computation_client->LookupAddressableDevice(context.device_id);
    case DLDeviceType::kDLCUDA:
      XLA_CHECK_EQ(computation_client->GetPlatformID(), xla::CudaId());
      return computation_client->LookupAddressableDevice(context.device_id);
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
        "Unknown/unsupported DLPack device type %d", context.device_type));
  }
}

// Reference: https://github.com/openxla/xla/blob/main/xla/python/dlpack.cc
absl::StatusOr<xla::PrimitiveType> DLDataTypeToPrimitiveType(DLDataType type) {
  if (type.lanes != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
      "DLPack types with lanes != 1 not implemented, got %d", type.lanes));
  }
  switch (type.code) {
    case kDLBool:
      switch (type.bits) {
        case 8:
          return xla::PrimitiveType::PRED;
        default:
          return absl::InvalidArgumentError(absl::StrFormat(
            "Only 8-bit DLPack booleans are supported, got %d bits",
            type.bits));
      }
    case kDLInt:
      switch (type.bits) {
        case 8:
          return xla::PrimitiveType::S8;
        case 16:
          return xla::PrimitiveType::S16;
        case 32:
          return xla::PrimitiveType::S32;
        case 64:
          return xla::PrimitiveType::S64;
        default:
          return absl::InvalidArgumentError(absl::StrFormat(
            "Invalid or unsupported DLPack integer width: %d bits", type.bits));
      }
    case kDLUInt:
      switch (type.bits) {
        case 8:
          return xla::PrimitiveType::U8;
        case 16:
          return xla::PrimitiveType::U16;
        case 32:
          return xla::PrimitiveType::U32;
        case 64:
          return xla::PrimitiveType::U64;
        default:
          return absl::InvalidArgumentError(absl::StrFormat(
            "Invalid or unsupported DLPack unsigned integer width: %d bits",
            type.bits));
      }
    case kDLFloat:
      switch (type.bits) {
        case 16:
          return xla::PrimitiveType::F16;
        case 32:
          return xla::PrimitiveType::F32;
        case 64:
          return xla::PrimitiveType::F64;
        default:
          return absl::InvalidArgumentError(absl::StrFormat(
            "Invalid or unsupported DLPack float width: %d bits", type.bits));
      }
    case kDLBfloat:
      switch (type.bits) {
        case 16:
          return xla::PrimitiveType::BF16;
        default:
          return absl::InvalidArgumentError(absl::StrFormat(
            "Invalid or unsupported DLPack Bfloat width: %d bits", type.bits));
      }
    case kDLComplex:
      switch (type.bits) {
        case 64:
          return xla::PrimitiveType::C64;
        case 128:
          return xla::PrimitiveType::C128;
        default:
          return absl::InvalidArgumentError(absl::StrFormat(
            "Invalid or unsupported DLPack complex width: %d bits", type.bits));
      }
    default:
      return absl::InvalidArgumentError(
        absl::StrFormat("Unknown or invalid DLPack type code %d", type.code));
  }
}

// Reference: https://github.com/openxla/xla/blob/main/xla/python/dlpack.cc
absl::StatusOr<std::vector<int64_t>> StridesToLayout(
  absl::Span<int64_t const> dims, absl::Span<int64_t const> strides) {
  XLA_CHECK_EQ(dims.size(), strides.size());
  std::vector<int64_t> minor_to_major(dims.size());
  std::iota(minor_to_major.begin(), minor_to_major.end(), 0);
  absl::c_sort(minor_to_major, [&](int a, int b) {
    if (strides[a] < strides[b]) {
      return true;
    }
    if (strides[a] > strides[b]) {
      return false;
    }
    // If two dimensions have the same stride, prefer the major-to-minor
    // interpretation of the ordering, since that's what JAX wants.
    return b < a;
  });
  int64_t stride = 1;
  for (int64_t d : minor_to_major) {
    if (dims[d] > 1 && strides[d] != stride) {
      return absl::InvalidArgumentError(absl::StrFormat(
        "Only DLPack tensors with trivial (compact) striding are supported; "
        "i.e., tensors whose striding represents a transposition of the "
        "underlying buffer but not broadcasting. Dimensions were: [%s], "
        "strides were [%s].",
        absl::StrJoin(dims, ","), absl::StrJoin(strides, ",")));
    }
    stride *= dims[d];
  }
  return minor_to_major;
}

runtime::ComputationClient::DataPtr FromDLPack(DLManagedTensor* dlmt) {
  XLA_CHECK(dlmt->dl_tensor.ndim >= 0)
    << "Number of dimensions in DLManagedTensor must be nonnegative, got "
    << dlmt->dl_tensor.ndim;
  auto computation_client_status = runtime::GetComputationClient();
  XLA_CHECK_OK(computation_client_status.status())
    << "Failed to get computation client.";
  auto computation_client = computation_client_status.value();
  xla::PjRtDevice* device =
    DeviceForDLDevice(dlmt->dl_tensor.device, computation_client).value();
  absl::Span<int64_t const> dimensions(
    const_cast<int64_t*>(dlmt->dl_tensor.shape), dlmt->dl_tensor.ndim);
  xla::PrimitiveType element_type =
    DLDataTypeToPrimitiveType(dlmt->dl_tensor.dtype).value();

  std::vector<int64_t> minor_to_major;
  if (
    dlmt->dl_tensor.strides
    && absl::c_find(dimensions, 0) == dimensions.end()) {
    absl::Span<int64_t const> strides(
      const_cast<int64_t*>(dlmt->dl_tensor.strides), dlmt->dl_tensor.ndim);
    minor_to_major = StridesToLayout(dimensions, strides).value();
  } else {
    minor_to_major.resize(dlmt->dl_tensor.ndim);
    std::iota(minor_to_major.rbegin(), minor_to_major.rend(), 0);
  }
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
    element_type, dimensions, minor_to_major);

  std::function<void()> on_delete_callback;
  if (dlmt->deleter) {
    on_delete_callback = [dlmt, deleter = dlmt->deleter]() { deleter(dlmt); };
    dlmt->deleter = nullptr;
  }
  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> pjrt_buffer =
    device->client()->CreateViewOfDeviceBuffer(
      static_cast<char*>(dlmt->dl_tensor.data) + dlmt->dl_tensor.byte_offset,
      shape, *device->default_memory_space(), on_delete_callback);
  XLA_CHECK_OK(pjrt_buffer.status()) << "Failed to create a pjrt buffer.";
  XLA_CHECK(pjrt_buffer.value() != nullptr) << "pjrt buffer is null.";

  runtime::ComputationClient::DataPtr data =
    runtime::PjRtComputationClient::CreateData(
      computation_client->PjRtDeviceToString(device), shape,
      std::move(pjrt_buffer.value()));

  return data;
}

}  // namespace xla_launcher
