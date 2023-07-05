/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_ACTIVITY_NOISE_FILTER_ALGORITHM_H
#define METAVISION_SDK_CV_ACTIVITY_NOISE_FILTER_ALGORITHM_H

#include <memory>
#include <sstream>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/core/algorithms/detail/internal_algorithms.h"
#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/base/events/event2d.h"

namespace Metavision {

/// @brief Filter that accepts events if a similar event has happened during a certain time window in the past,
/// in the neighborhood of its coordinates.
template<typename Use64Bits = std::true_type>
class ActivityNoiseFilterAlgorithm {
    using timestamp_32 = std::int32_t;

public:
    /// @brief Builds a new ActivityNoiseFilterAlgorithm object
    /// @param width Maximum X coordinate of the events in the stream
    /// @param height Maximum Y coordinate of the events in the stream
    /// @param threshold Length of the time window for activity filtering (in us)
    inline ActivityNoiseFilterAlgorithm(std::uint32_t width, std::uint32_t height, timestamp threshold);

    /// @brief Default destructor
    ~ActivityNoiseFilterAlgorithm() = default;

    /// @brief Applies the Activity Noise filter to the given input buffer storing the result in the output buffer
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @tparam OutputIt Read-Write output event iterator type. Works for iterators over containers of @ref EventCD
    /// or equivalent
    /// @param it_begin Iterator to first input event
    /// @param it_end Iterator to the past-the-end event
    /// @param inserter Output iterator or back inserter
    /// @return Iterator pointing to the past-the-end event added in the output
    template<class InputIt, class OutputIt>
    inline OutputIt process_events(InputIt it_begin, InputIt it_end, OutputIt inserter) {
#ifdef __AVX2__
        return test(it_begin, it_end, inserter, Use64Bits{});
#else
        return detail::insert_if(it_begin, it_end, inserter, std::ref(*this));
#endif
    }

    /// @brief Returns the current threshold
    /// @return Threshold applied while filtering
    inline timestamp get_threshold();

    /// @brief Sets the threshold
    /// @param threshold threshold
    inline void set_threshold(timestamp threshold);
#ifdef __AVX2__

    /// @brief Checks if the event is filtered
    /// @return true if the event is filtered
    template<class InputIt, class OutputIt>
    inline OutputIt test(InputIt first, InputIt last, OutputIt inserter, std::true_type const &);

    /// @brief Checks if the event is filtered
    /// @return true if the event is filtered
    template<class InputIt, class OutputIt>
    inline OutputIt test(InputIt first, InputIt last, OutputIt inserter, std::false_type const &);
#else
    /// @brief Checks if the event is filtered
    /// @return true if the event is filtered
    inline bool operator()(const Event2d &event);
#endif

private:
    std::uint32_t width_  = 0;
    std::uint32_t height_ = 0;
    std::uint32_t pixels_ = 0;
#ifndef __AVX2__
    std::uint32_t w_minus_2;
#endif
    timestamp thres_{};
    std::vector<timestamp> last_ts_{};
    std::vector<timestamp_32> last_ts_32_{};
};

template<typename Use64Bits>
inline ActivityNoiseFilterAlgorithm<Use64Bits>::ActivityNoiseFilterAlgorithm(std::uint32_t width, std::uint32_t height,
                                                                             timestamp threshold) :
    width_(width + 2),
    height_(height + 2),
    pixels_(width_ * height_),
#ifndef __AVX2__
    w_minus_2(width_ - 2),
#endif
    thres_(threshold) {
    // 4 is added here as the avx2 version will load past the end if the event occurs at the last pixel
    if (Use64Bits{}) {
        last_ts_.resize(pixels_ + 4, static_cast<int>(-2 * thres_));
    } else {
        last_ts_32_.resize(pixels_ + 4, static_cast<int>(-2 * thres_));
    }
}

template<typename Use64Bits>
inline timestamp ActivityNoiseFilterAlgorithm<Use64Bits>::get_threshold() {
    return thres_;
}

template<typename Use64Bits>
inline void ActivityNoiseFilterAlgorithm<Use64Bits>::set_threshold(timestamp threshold) {
    thres_ = threshold;
}

#ifdef __AVX2__
template<typename Use64Bits>
template<class InputIt, class OutputIt>
inline OutputIt ActivityNoiseFilterAlgorithm<Use64Bits>::test(InputIt first, InputIt last, OutputIt inserter,
                                                              std::true_type const &) {
    __m256i v_mask        = _mm256_set_epi64x(0x0lu, 0xFFFFFFFFFFFFFFFFlu, 0xFFFFFFFFFFFFFFFFlu, 0xFFFFFFFFFFFFFFFFlu);
    __m256i v_mask_middle = _mm256_set_epi64x(0x0, 0xFFFFFFFFFFFFFFFFlu, 0x0lu, 0xFFFFFFFFFFFFFFFFlu);
    for (auto it = first; it != last; ++it) {
        const Event2d &event        = *it;
        timestamp ts                = event.t;
        auto index                  = (event.y) * width_ + (event.x);
        const timestamp timecompare = ts - thres_;
        timestamp *ptr              = &last_ts_[0] + index;
        index += width_ + 1;
        __m256i first       = _mm256_loadu_si256((__m256i *)ptr);
        __m256i second      = _mm256_loadu_si256((__m256i *)(ptr + width_));
        __m256i third       = _mm256_loadu_si256((__m256i *)(ptr + 2 * width_));
        __m256i v_time_comp = _mm256_set1_epi64x(timecompare - 1);
        __m256i v_comp0     = _mm256_cmpgt_epi64(first, v_time_comp);
        __m256i v_comp1     = _mm256_cmpgt_epi64(second, v_time_comp);
        v_comp1             = _mm256_and_si256(v_comp1, v_mask_middle);
        __m256i v_comp2     = _mm256_cmpgt_epi64(third, v_time_comp);
        __m256i v_res       = _mm256_or_si256(_mm256_or_si256(v_comp0, v_comp1), v_comp2);
        __m256i v_res_mask  = _mm256_and_si256(v_res, v_mask);
        int ires            = (_mm256_testc_si256(_mm256_setzero_si256(), v_res_mask)) == 0;
        last_ts_[index]     = ts;
        if (ires) {
            *inserter = *it;
            ++inserter;
        }
    }
    return inserter;
}
template<typename Use64Bits>
template<class InputIt, class OutputIt>
inline OutputIt ActivityNoiseFilterAlgorithm<Use64Bits>::test(InputIt first, InputIt last, OutputIt inserter,
                                                              std::false_type const &) {
    using timestamp_32    = typename ActivityNoiseFilterAlgorithm::timestamp_32;
    __m128i v_mask        = _mm_set_epi32(0x0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    __m128i v_mask_middle = _mm_set_epi32(0x0, 0xFFFFFFFF, 0x0, 0xFFFFFFFF);
    for (auto it = first; it != last; ++it) {
        const Event2d &event           = *it;
        timestamp_32 ts                = event.t;
        auto index                     = ((event.y) * width_) + (event.x);
        const timestamp_32 timecompare = ts - thres_;
        timestamp_32 *ptr              = &last_ts_32_[0] + index;
        index += width_ + 1;
        __m128i first       = _mm_loadu_si128((__m128i *)ptr);
        __m128i second      = _mm_loadu_si128((__m128i *)(ptr + width_));
        __m128i third       = _mm_loadu_si128((__m128i *)(ptr + 2 * width_));
        __m128i v_time_comp = _mm_set1_epi32(timecompare - 1);
        __m128i v_comp0     = _mm_cmpgt_epi32(first, v_time_comp);
        __m128i v_comp1     = _mm_cmpgt_epi32(second, v_time_comp);
        v_comp1             = _mm_and_si128(v_comp1, v_mask_middle);
        __m128i v_comp2     = _mm_cmpgt_epi32(third, v_time_comp);
        __m128i v_res       = _mm_or_si128(_mm_or_si128(v_comp0, v_comp1), v_comp2);
        __m128i v_res_mask  = _mm_and_si128(v_res, v_mask);
        int ires            = (_mm_testc_si128(_mm_setzero_si128(), v_res_mask)) == 0;
        last_ts_32_[index]  = ts;
        if (ires) {
            *inserter = *it;
            ++inserter;
        }
    }
    return inserter;
}
#else
template<typename Use64Bits>
inline bool ActivityNoiseFilterAlgorithm<Use64Bits>::operator()(const Event2d &event) {
    const auto index = (event.y + 1) * width_ + (event.x + 1);
    const auto timecompare = event.t - thres_;
    last_ts_[index] = event.t;
    auto *ptr = &last_ts_[0] + index - 1;
    return *ptr >= timecompare || *(ptr += 2) >= timecompare || *(ptr += w_minus_2) >= timecompare ||
           *(++ptr) >= timecompare || *(++ptr) >= timecompare || *(ptr -= (width_ * 2 + 2)) >= timecompare ||
           *(++ptr) >= timecompare || *(++ptr) >= timecompare;

    // For efficiency, we do it using pointers. It's equivalent:
    //    if ( last_ts_[index-width_ -1] >= timecompare ||
    //         last_ts_[index-width_ ]   >= timecompare ||
    //         last_ts_[index-width_ +1] >= timecompare ||
    //         last_ts_[index-1]         >= timecompare ||
    //         last_ts_[index+1]         >= timecompare ||
    //         last_ts_[index+width_ -1] >= timecompare ||
    //         last_ts_[index+width_ ]   >= timecompare ||
    //         last_ts_[index+width_ +1] >= timecompare) {
}
#endif
} // namespace Metavision
#endif // METAVISION_SDK_CV_ACTIVITY_NOISE_FILTER_ALGORITHM_H
