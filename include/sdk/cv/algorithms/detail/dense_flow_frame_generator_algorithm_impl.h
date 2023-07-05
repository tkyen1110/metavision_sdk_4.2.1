/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

namespace Metavision {

template<typename EventIt>
void DenseFlowFrameGeneratorAlgorithm::process_events(EventIt it_begin, EventIt it_end) {
    for (auto it = it_begin; it != it_end; ++it) {
        auto &px = states_[it->y * width_ + it->x];
        px.accumulate(*it, policy_, vx_.at<float>(it->y, it->x), vy_.at<float>(it->y, it->x));
    }
}

} // namespace Metavision
