#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

#include "input_interface/transport_freshness.hpp"

namespace {
using Subscriber = ZMQPackedMessageSubscriber;

Subscriber::DecodedHeader MetadataHeader() {
  Subscriber::DecodedHeader header;
  if (!Subscriber::DecodeHeaderJSON(
      R"({"v":1,"endian":"le","fields":[{"name":"epoch","dtype":"i64","shape":[1]},{"name":"sequence","dtype":"i64","shape":[1]},{"name":"sent_unix_us","dtype":"i64","shape":[1]}]})",
      header)) throw std::runtime_error("test header did not parse");
  return header;
}

std::vector<Subscriber::BufferView> Views(const std::array<int64_t, 3>& values) {
  return {{&values[0], sizeof(int64_t)}, {&values[1], sizeof(int64_t)},
          {&values[2], sizeof(int64_t)}};
}
}  // namespace

TEST(PackedTransport, RejectsMalformedHeaders) {
  Subscriber::DecodedHeader header;
  EXPECT_FALSE(Subscriber::DecodeHeaderJSON(
      R"({"v":1,"endian":"le","fields":[{"name":"x","dtype":"wat","shape":[1]}]})",
      header));
  EXPECT_FALSE(Subscriber::DecodeHeaderJSON(
      R"({"v":1,"endian":"le","fields":[{"name":"x","dtype":"f32","shape":[1]},{"name":"x","dtype":"f32","shape":[1]}]})",
      header));
  EXPECT_FALSE(Subscriber::DecodeHeaderJSON(
      R"({"v":1,"endian":"middle","fields":[{"name":"x","dtype":"f32","shape":[1]}]})",
      header));
}

TEST(PackedTransport, RejectsStaleDuplicateAndRegressingMessages) {
  auto header = MetadataHeader();
  TransportFreshnessGuard guard;
  const int64_t now = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  const std::array<int64_t, 3> first{1, 1, now};
  EXPECT_TRUE(guard.Accept(header, Views(first)));
  EXPECT_FALSE(guard.Accept(header, Views(first)));
  const std::array<int64_t, 3> stale{1, 2, now - 1000000};
  EXPECT_FALSE(guard.Accept(header, Views(stale)));
  const std::array<int64_t, 3> newer{2, 0, now};
  EXPECT_TRUE(guard.Accept(header, Views(newer)));
  const std::array<int64_t, 3> regressed{1, 99, now};
  EXPECT_FALSE(guard.Accept(header, Views(regressed)));
}
