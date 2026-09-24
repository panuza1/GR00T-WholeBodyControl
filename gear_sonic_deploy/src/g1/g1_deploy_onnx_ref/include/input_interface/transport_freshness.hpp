#ifndef TRANSPORT_FRESHNESS_HPP
#define TRANSPORT_FRESHNESS_HPP

#include "zmq_packed_message_subscriber.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

class TransportFreshnessGuard {
 public:
  explicit TransportFreshnessGuard(int64_t max_age_us = 200000,
                                   int64_t max_future_us = 100000)
      : max_age_us_(max_age_us), max_future_us_(max_future_us) {}

  bool Accept(const ZMQPackedMessageSubscriber::DecodedHeader& header,
              const std::vector<ZMQPackedMessageSubscriber::BufferView>& buffers) {
    int64_t epoch = 0, sequence = 0, sent_unix_us = 0;
    const bool has_epoch = ReadI64(header, buffers, "epoch", epoch);
    const bool has_sequence = ReadI64(header, buffers, "sequence", sequence);
    const bool has_sent = ReadI64(header, buffers, "sent_unix_us", sent_unix_us);
    if (!has_epoch && !has_sequence && !has_sent) return true;  // Legacy producers.
    if (!has_epoch || !has_sequence || !has_sent || epoch < 0 || sequence < 0) return false;

    const int64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const int64_t age_us = now_us - sent_unix_us;
    if (age_us > max_age_us_ || age_us < -max_future_us_) return false;
    if (initialized_ && (epoch < epoch_ || (epoch == epoch_ && sequence <= sequence_))) return false;
    epoch_ = epoch;
    sequence_ = sequence;
    initialized_ = true;
    return true;
  }

 private:
  static bool ReadI64(const ZMQPackedMessageSubscriber::DecodedHeader& header,
                      const std::vector<ZMQPackedMessageSubscriber::BufferView>& buffers,
                      const std::string& name, int64_t& value) {
    if (header.fields.size() != buffers.size()) return false;
    for (size_t i = 0; i < header.fields.size(); ++i) {
      if (header.fields[i].name != name) continue;
      if (header.fields[i].dtype != "i64" || buffers[i].size != sizeof(value)) return false;
      std::memcpy(&value, buffers[i].data, sizeof(value));
      if (header.NeedsByteSwap()) value = byte_swap(value);
      return true;
    }
    return false;
  }

  int64_t max_age_us_;
  int64_t max_future_us_;
  int64_t epoch_ = -1;
  int64_t sequence_ = -1;
  bool initialized_ = false;
};

#endif
