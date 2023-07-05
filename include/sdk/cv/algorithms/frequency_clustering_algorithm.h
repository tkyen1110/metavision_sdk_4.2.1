/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_FREQUENCY_CLUSTERING_ALGORITHM_H
#define METAVISION_SDK_CV_FREQUENCY_CLUSTERING_ALGORITHM_H

#include <vector>
#include <memory>

#include "metavision/sdk/cv/configs/frequency_clustering_algorithm_config.h"
#include "metavision/sdk/cv/events/event_frequency.h"
#include "metavision/sdk/cv/events/event_frequency_cluster.h"

namespace Metavision {

class FrequencyClusteringAlgorithmInternal;

/// @brief Frequency clustering algorithm. Processes input frequency events and groups them in clusters.
class FrequencyClusteringAlgorithm {
public:
    using frequency_precision = float;
    using InputEventType      = Metavision::Event2dFrequency<frequency_precision>;
    using OutputEventType     = Metavision::Event2dFrequencyCluster<frequency_precision>;

    /// @brief Builds a new FrequencyClusteringAlgorithm object
    /// @param width Sensor's width
    /// @param height Sensor's height
    /// @param config frequency clustering algorithm's configuration
    FrequencyClusteringAlgorithm(int width, int height, const FrequencyClusteringAlgorithmConfig &config);
    /// @brief Destructor
    ~FrequencyClusteringAlgorithm();

    /// @brief Updates clusters from input events
    /// @tparam InputIt Read-Only input event iterator type. Works for iterators over buffers of @ref EventCD
    /// or equivalent
    /// @tparam OutputIt Read-Write output clusters iterator type.
    /// @param it_begin Iterator to first input event
    /// @param it_end Iterator to the past-the-end event
    /// @param inserter Output iterator or back inserter to output clusters.
    template<class InputIt, class OutputIt>
    void process_events(InputIt it_begin, InputIt it_end, OutputIt inserter);

private:
    std::unique_ptr<FrequencyClusteringAlgorithmInternal> algo_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_FREQUENCY_CLUSTERING_ALGORITHM_H
