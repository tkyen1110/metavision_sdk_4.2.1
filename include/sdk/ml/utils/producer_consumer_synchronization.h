/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ML_UTILS_PRODUCER_CONSUMER_SYNCHRONIZATION_H
#define METAVISION_SDK_ML_UTILS_PRODUCER_CONSUMER_SYNCHRONIZATION_H

#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <assert.h>

namespace Metavision {

/// @brief Synchronizes one producer with a consumer
///
/// The producer checks if the consumer is ready (previous work is finished),
/// it prepares the data for the consumer, and signals the consumer.
///
/// On its side, the consumer waits for data to be computed and applies a function
/// as soon as the data is ready
struct ProducerConsumerSynchronization {
    /// @brief Creates a ProducerConsumerSynchronization object
    ProducerConsumerSynchronization() : data_ready_(false), data_processed_(false) {}

    std::condition_variable signaller_;
    std::mutex lock_;
    bool data_ready_;
    bool data_processed_;

    /// @brief Sets data available for the consumer
    /// @warning This function should be called by the producer only
    void data_is_ready() {
        std::lock_guard<std::mutex> lk(lock_);
        data_ready_     = true;
        data_processed_ = false;
    }

    /// @brief Waits for the consumer to be ready (should be called only by the producer)
    /// @param done Boolean to release the condition. If done become True, the producer exits the function
    void producer_wait(std::atomic<bool> &done) {
        std::unique_lock<std::mutex> lk(lock_);
        signaller_.wait(lk, [this, &done] { return this->data_processed_ || done; });
    }
    /// @brief Waits for data to be computed
    /// @param done Boolean to release the condition. If done become True, the producer exits the function
    /// @param lambda_function Function to be called on data provided by the producer
    template<typename L>
    void consumer_wait(std::atomic<bool> &done, L &&lambda_function) {
        std::unique_lock<std::mutex> lk(lock_);
        if (!this->data_ready_) {
            signaller_.wait(lk, [this, &done] { return this->data_ready_ || done; });
        }
        assert(this->data_ready_ || done);
        lambda_function();
        data_ready_     = false;
        data_processed_ = true;
        lk.unlock();
        signaller_.notify_one();
    }

    /// @brief Notifies either producer or consumer to wake it up
    void notify() {
        signaller_.notify_one();
    }
};

} // namespace Metavision
#endif // METAVISION_SDK_ML_UTILS_PRODUCER_CONSUMER_SYNCHRONIZATION_H
