/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_MAP_GENERATOR_ASYNC_ALGORITHM_H
#define METAVISION_SDK_ANALYTICS_MAP_GENERATOR_ASYNC_ALGORITHM_H

#include <functional>
#include <opencv2/core.hpp>

#include "metavision/sdk/core/algorithms/async_algorithm.h"

namespace Metavision {

/// @brief Class that accumulates events in a map
///
/// More specifically, this class accumulates one of the event's fields in a map. Each map's cell contains the last
/// value of that field at that location. Under the hood this class uses the OpenCV's Mat_\<T\> structure, meaning that
/// it is only compatible with the types for which the Mat_\<T\> structure is compatible too (i.e. uchar, short, ushort,
/// int, float and double). This class is not compatible with OpenCV's vector types (i.e. instantiations of the
/// Vec\<T, int cn\> class).
///
/// @tparam T Type, one of the fields of which will be accumulated in the map
/// @tparam F Type of the field that will be accumulated in the map
/// @tparam f Address in @p T of the field to be accumulated
template<typename T, typename F, F T::*f>
class MapGeneratorAsyncAlgorithm : public AsyncAlgorithm<MapGeneratorAsyncAlgorithm<T, F, f>> {
public:
    friend class AsyncAlgorithm<MapGeneratorAsyncAlgorithm<T, F, f>>;

    using OutputMap = cv::Mat_<F>;
    using OutputCb  = std::function<void(timestamp, OutputMap &)>;

    /// @brief Builds a new @ref MapGeneratorAsyncAlgorithm object
    /// @param width Sensor's width (in pixels)
    /// @param height Sensor's height (in pixels)
    MapGeneratorAsyncAlgorithm(int width, int height);

    /// @brief Default destructor
    ~MapGeneratorAsyncAlgorithm() = default;

    /// @brief Sets the callback that will be called to let the user retrieve the generated map
    ///
    /// The generated map will be passed through the callback via a non constant reference, meaning that the client is
    /// free to copy it or swap it.
    ///
    /// @param cb The callback to call
    /// @note In case of a swap with a non initialized map, it will automatically be initialized by the @ref
    /// MapGeneratorAsyncAlgorithm.
    void set_output_callback(const OutputCb &cb);

private:
    /// @brief Function to process directly the events
    /// @tparam InputIt An iterator type over an event of type T having fields x, y, and t (i.e. in addition to the one
    /// given by the pointer to class' attribute in the template parameters)
    /// @param begin The first iterator of the @p T events buffer to process
    /// @param end The end iterator of the @p T events buffer to process
    template<typename InputIt>
    inline void process_online(InputIt begin, InputIt end);

    /// @brief Resets the output map
    ///
    /// It allocates it if needed and sets all the values to 0.
    void reset_map();

    /// @brief Function to process the state that is called every n_events or n_us
    void process_async(const timestamp processing_ts, const size_t n_processed_events);

    int width_;
    int height_;
    OutputCb output_cb_;
    cv::Mat_<F> map_;
};

template<typename T, typename F, F T::*f>
MapGeneratorAsyncAlgorithm<T, F, f>::MapGeneratorAsyncAlgorithm(int width, int height) {
    width_     = width;
    height_    = height;
    output_cb_ = [](timestamp, OutputMap &) {};

    reset_map();
}

template<typename T, typename F, F T::*f>
void MapGeneratorAsyncAlgorithm<T, F, f>::set_output_callback(const OutputCb &cb) {
    output_cb_ = cb;
}

template<typename T, typename F, F T::*f>
template<typename InputIt>
void MapGeneratorAsyncAlgorithm<T, F, f>::process_online(InputIt begin, InputIt end) {
    for (auto it = begin; it != end; ++it)
        map_.template at<F>(it->y, it->x) = (*it).*f;
}

template<typename T, typename F, F T::*f>
void MapGeneratorAsyncAlgorithm<T, F, f>::reset_map() {
    map_.create(height_, width_);
    map_.setTo(static_cast<F>(0));
}

template<typename T, typename F, F T::*f>
void MapGeneratorAsyncAlgorithm<T, F, f>::process_async(const timestamp processing_ts,
                                                        const size_t n_processed_events) {
    output_cb_(processing_ts, map_);

    reset_map();
}

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_MAP_GENERATOR_ASYNC_ALGORITHM_H
