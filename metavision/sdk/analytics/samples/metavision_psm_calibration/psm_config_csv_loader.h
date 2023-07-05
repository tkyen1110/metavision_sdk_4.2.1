/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef PSM_CALIB_CSV_LOADER_H
#define PSM_CALIB_CSV_LOADER_H

#include <array>
#include <string>
#include <boost/variant.hpp>
#include <metavision/sdk/analytics/configs/line_cluster_tracking_config.h>
#include <metavision/sdk/analytics/configs/line_particle_tracking_config.h>

namespace Metavision {

/// @brief Class that suggests optimal PsmAlgorithm parameters and dumps them into a CSV, so that the user is free to
/// tune them and reload the parameters as many times as needed
///
/// The class uses a heuristic based on the lines of interest and the average particle speed and size to initialize
/// optimal PsmAlgorithm parameters. It's always possible to overwrite current CSV with the initial suggested
/// PsmAlgorithm configuration
///
/// Each line of the CSV config file will looks like below:
///    20, 	Precision Time // [16; 24]
///
/// The line is split by "," into two parts. The left part is the selected value and the right part specifies the name
/// of the variable and an interval of possible values. This interval is also given by the heuristic
class PsmConfigCsvLoader {
public:
    /// @brief Constructor
    /// @param width Sensor's width (in pixels)
    /// @param height Sensor's height (in pixels)
    /// @param directory_path Directory used to save the CSV config file
    /// @param avg_speed_pix_ms Average particle speed in pix/ms
    /// @param avg_size_pix Approximate particle size in pix
    /// @param is_going_down Specify if the particles are going downwards or upwards
    /// @param min_y_line Lowest lines of interest's ordinate
    /// @param max_y_line Largest lines of interest's ordinate
    /// @param num_lines Number of lines of interest
    /// @throw std::invalid_argument if it failed to create @p directory_path
    PsmConfigCsvLoader(int width, int height, const std::string &directory_path, int avg_speed_pix_ms, int avg_size_pix,
                       bool is_going_down, int min_y_line, int max_y_line, int num_lines);

    /// @brief Gets PsmAlgorithm configuration from CSV
    /// @param cluster_config Ouput line cluster configuration
    /// @param particle_config Ouput line particle configuration
    /// @param num_process_before_matching Ouput number of process before matching
    bool load_config(LineClusterTrackingConfig &cluster_config, LineParticleTrackingConfig &particle_config,
                     int &num_process_before_matching);

    /// @brief Overwrites current CSV with the initial suggested PsmAlgorithm configuration
    void reset_config();

private:
    struct CSVParameter {
        using Value = boost::variant<unsigned int, float>;

        std::string desc;
        Value value;
    };

    template<typename T>
    void write_param(PsmConfigCsvLoader::CSVParameter &p, const std::string &name, const T &min, const T &value,
                     const T &max, const std::string &additional_info = "");

    std::string config_csv_path_;              ///< Path to the CSV config file
    std::array<CSVParameter, 11> init_params_; ///< Initial suggested PsmAlgorithm configuration
    const bool is_going_down_;                 ///< Set to true for downward motions, and false for upward motions
};

} // namespace Metavision

#endif // PSM_CALIB_CSV_LOADER_H
