/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ML_DETAIL_CD_PROCESSING_RESCALING_TYPE_H
#define METAVISION_SDK_ML_DETAIL_CD_PROCESSING_RESCALING_TYPE_H

namespace Metavision {

/// @brief Selects the rescaling algorithm depending on the required conversion
enum class CDProcessingRescalingType { None, Downscaling, Upscaling };

} // namespace Metavision

#endif // METAVISION_SDK_ML_DETAIL_CD_PROCESSING_RESCALING_TYPE_H
