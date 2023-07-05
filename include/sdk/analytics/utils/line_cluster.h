/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_LINE_CLUSTER_H
#define METAVISION_SDK_ANALYTICS_LINE_CLUSTER_H

#include <cstddef>

namespace Metavision {

/// @brief Structure that stores the begin and end position of a cluster in a line
struct LineCluster {
    LineCluster() = default;
    LineCluster(int x_begin, int x_end) : x_begin(x_begin), x_end(x_end) {}

    int x_begin = -1;
    int x_end   = -2;
};

/// @brief Structure representing a 1D Cluster with its id
struct LineClusterWithId {
    LineClusterWithId() = default;
    LineClusterWithId(int x_begin, int x_end, size_t id) : x_begin(x_begin), x_end(x_end), id(id) {}

    int x_begin = -1;
    int x_end   = -2;
    size_t id   = -1;
};

/// @brief Structure representing a 1D cluster with its timestamp
struct TimedLineCluster {
    TimedLineCluster() = default;
    TimedLineCluster(int x_begin, int x_end, int ts) : x_begin(x_begin), x_end(x_end), ts(ts) {}

    int x_begin = -1;
    int x_end   = -2;
    int ts      = -1;
};

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_LINE_CLUSTER_H
