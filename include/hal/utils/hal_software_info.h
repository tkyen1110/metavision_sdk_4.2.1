/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A.                                                                                       *
 *                                                                                                                    *
 * Licensed under the Apache License, Version 2.0 (the "License");                                                    *
 * you may not use this file except in compliance with the License.                                                   *
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0                                 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed   *
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                      *
 * See the License for the specific language governing permissions and limitations under the License.                 *
 **********************************************************************************************************************/

#ifndef METAVISION_HAL_HAL_SOFTWARE_INFO_H
#define METAVISION_HAL_HAL_SOFTWARE_INFO_H

#include <string>

#include "metavision/sdk/base/utils/software_info.h"

namespace Metavision {

/// @brief Returns various software information, such as the version, about the Metavision HAL used at run time
/// @return Software information
Metavision::SoftwareInfo &get_hal_software_info();

} // namespace Metavision

#endif // METAVISION_HAL_HAL_SOFTWARE_INFO_H
