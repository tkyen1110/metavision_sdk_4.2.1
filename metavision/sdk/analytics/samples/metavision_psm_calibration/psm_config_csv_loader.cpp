/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include <boost/filesystem.hpp>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <metavision/sdk/base/utils/log.h>

#include "psm_config_csv_loader.h"

namespace bfs = boost::filesystem;

namespace Metavision {

static constexpr char kCsvDelimiter = ',';

static constexpr float kPrecisionTimeMinRatio = 0.8f;
static constexpr float kPrecisionTimeMaxRatio = 1.2f;

static constexpr unsigned int kBitsetSizeMin = 4;
static constexpr unsigned int kBitsetSize    = 4;
static constexpr unsigned int kBitsetSizeMax = 7;

static constexpr unsigned int kClusterThsMin = 1;
static constexpr unsigned int kClusterThsVal = 3;
static constexpr unsigned int kClusterThsMax = 4;

static constexpr float kNumClusterThsMinRatio = 0.5f;
static constexpr float kNumClusterThsValRatio = 0.7f;
static constexpr float kNumClusterThsMaxRatio = 0.8f;

static constexpr unsigned int kMinInterClusterDistMin = 1;
static constexpr unsigned int kMinInterClusterDistVal = 1;
static constexpr unsigned int kMinInterClusterDistMax = 10;

static constexpr float kLearningRateMin = 0.6f;
static constexpr float kLearningRateVal = 0.9f;
static constexpr float kLearningRateMax = 1.0f;

static constexpr float kClampingMin = 4.f;
static constexpr float kClampingVal = 5.f;
static constexpr float kClampingMax = 10.f;

static constexpr unsigned int kMaxAngleThsMin = 30;
static constexpr unsigned int kMaxAngleThsVal = 45;
static constexpr unsigned int kMaxAngleThsMax = 60;

static constexpr float kParticleMatchThsMin = 0.4f;
static constexpr float kParticleMatchThsVal = 0.5f;
static constexpr float kParticleMatchThsMax = 0.9f;

static constexpr unsigned int kNumProcessBeforeMatchMin = 1;
static constexpr float kNumProcessBeforeMatchValRatio   = 0.8f;
static constexpr float kNumProcessBeforeMatchMaxRatio   = 2.f;

namespace details {
template<typename T>
T read_value(const std::string &s);

template<>
unsigned int read_value<unsigned int>(const std::string &s) {
    return static_cast<unsigned int>(std::stoul(s));
}

template<>
float read_value<float>(const std::string &s) {
    return std::stof(s);
}

template<typename T>
std::string type_to_str();

template<>
std::string type_to_str<unsigned int>() {
    return "unsigned int";
}

template<>
std::string type_to_str<float>() {
    return "float";
}

template<typename T>
T read_param(std::ifstream &file) {
    std::string line, field;
    std::getline(file, line);
    std::istringstream ss(line);
    std::getline(ss, field, kCsvDelimiter);
    return details::read_value<T>(field);
}
} // namespace details

PsmConfigCsvLoader::PsmConfigCsvLoader(int width, int height, const std::string &directory_path, int avg_speed_pix_ms,
                                       int avg_size_pix, bool is_going_down, int min_y_line, int max_y_line,
                                       int num_lines) :
    is_going_down_(is_going_down) {
    // Create Directory if needed
    if (!bfs::exists(directory_path)) {
        try {
            bfs::create_directories(directory_path);
        } catch (bfs::filesystem_error &e) { throw std::invalid_argument(e.what()); }
    }

    // Define CSV config file path
    config_csv_path_ = (bfs::path(directory_path) / bfs::path("psm_test_config.csv")).string();

    // Initialize PSM algorithm parameters (LineClusterTrackingConfig)
    const auto precision_time_us = static_cast<unsigned int>(std::round(1000.f / avg_speed_pix_ms));
    const auto precision_time_us_min =
        static_cast<unsigned int>(std::round(kPrecisionTimeMinRatio * precision_time_us));
    const auto precision_time_us_max =
        static_cast<unsigned int>(std::round(kPrecisionTimeMaxRatio * precision_time_us));
    write_param(init_params_[0], "Precision Time", precision_time_us_min, precision_time_us, precision_time_us_max);
    write_param(init_params_[1], "Bitset Buffer Size", kBitsetSizeMin, kBitsetSize, kBitsetSizeMax);
    write_param(init_params_[2], "Cluster Threshold", kClusterThsMin, kClusterThsVal, kClusterThsMax);

    const auto avg_num_triggered_clusters =
        static_cast<unsigned int>(std::round((avg_size_pix * precision_time_us * avg_speed_pix_ms) / 1000.f));
    const auto num_triggered_clusters =
        static_cast<unsigned int>(std::round(avg_num_triggered_clusters * kNumClusterThsValRatio));
    const auto min_num_triggered_clusters =
        static_cast<unsigned int>(std::round(kNumClusterThsMinRatio * avg_num_triggered_clusters));
    const auto max_num_triggered_clusters =
        static_cast<unsigned int>(std::round(avg_num_triggered_clusters * kNumClusterThsMaxRatio));

    write_param(init_params_[3], "Number of clusters Threshold", min_num_triggered_clusters, num_triggered_clusters,
                max_num_triggered_clusters);
    write_param(init_params_[4], "Min Inter Clusters Distance", kMinInterClusterDistMin, kMinInterClusterDistVal,
                kMinInterClusterDistMax);
    write_param(init_params_[5], "Learning Rate", kLearningRateMin, kLearningRateVal, kLearningRateMax);
    write_param(init_params_[6], "Clamping", kClampingMin, kClampingVal, kClampingMax, "or -1");

    // Initialize PSM algorithm parameters (LineParticleTrackingConfig)
    const int scope_rows           = (max_y_line - min_y_line);
    const int dy_rows              = (max_y_line - min_y_line) / num_lines;
    const auto dt_first_match_ths_ = static_cast<unsigned int>(std::round((scope_rows * 1000.f) / avg_speed_pix_ms));
    const auto dt_first_match_ths_min_ =
        static_cast<unsigned int>(std::round((2.f * dy_rows * 1000) / avg_speed_pix_ms));
    const auto dt_first_match_ths_max = static_cast<unsigned int>(std::round((height * 1000.f) / avg_speed_pix_ms));
    write_param(init_params_[7], "Dt First Match Threshold", dt_first_match_ths_min_, dt_first_match_ths_,
                dt_first_match_ths_max);

    write_param(init_params_[8], "Max Angle Threshold (Deg)", kMaxAngleThsMin, kMaxAngleThsVal, kMaxAngleThsMax);
    write_param(init_params_[9], "Particle Matching Threshold", kParticleMatchThsMin, kParticleMatchThsVal,
                kParticleMatchThsMax);

    const auto avg_num_process =
        static_cast<unsigned int>(std::round((dy_rows * precision_time_us * avg_speed_pix_ms) / 1000.f));
    const auto num_process = static_cast<unsigned int>(std::round(kNumProcessBeforeMatchValRatio * avg_num_process));
    const auto max_num_process =
        static_cast<unsigned int>(std::round(kNumProcessBeforeMatchMaxRatio * avg_num_process));
    write_param(init_params_[10], "Number of process before matching", kNumProcessBeforeMatchMin, num_process,
                max_num_process);

    // Dump initial configuration
    reset_config();
}

bool PsmConfigCsvLoader::load_config(LineClusterTrackingConfig &cluster_config,
                                     LineParticleTrackingConfig &particle_config, int &num_process_before_matching) {
    std::ifstream csv_file;
    csv_file.open(config_csv_path_);
    if (!csv_file.is_open()) {
        MV_LOG_ERROR() << "Configuration file" << config_csv_path_ << "couldn't be opened.";
        return false;
    }

    // Parse CSV
    try {
        // Update LineClusterTrackingConfig
        cluster_config.precision_time_us_           = details::read_param<unsigned int>(csv_file);
        cluster_config.bitsets_buffer_size_         = details::read_param<unsigned int>(csv_file);
        cluster_config.cluster_ths_                 = details::read_param<unsigned int>(csv_file);
        cluster_config.num_clusters_ths_            = details::read_param<unsigned int>(csv_file);
        cluster_config.min_inter_clusters_distance_ = details::read_param<unsigned int>(csv_file);
        cluster_config.learning_rate_               = details::read_param<float>(csv_file);
        cluster_config.max_dx_allowed_              = details::read_param<float>(csv_file);
        cluster_config.max_nbr_empty_rows_          = 0;

        // Update LineParticleTrackingConfig
        particle_config.dt_first_match_ths_ = details::read_param<unsigned int>(csv_file);
        particle_config.tan_angle_ths_      = std::tan(details::read_param<unsigned int>(csv_file) * 3.14 / 180.0);
        particle_config.matching_ths_       = details::read_param<float>(csv_file);
        particle_config.is_going_down_      = is_going_down_;

        // Update num_process_before_matching
        num_process_before_matching = details::read_param<unsigned int>(csv_file);

    } catch (const std::invalid_argument &ia) {
        MV_LOG_ERROR() << "Configuration file" << config_csv_path_ << "is not valid.";
        MV_LOG_ERROR() << ia.what();
        return false;
    }

    return true;
}

void PsmConfigCsvLoader::reset_config() {
    std::ofstream myfile;
    myfile.open(config_csv_path_);
    if (!myfile.is_open()) {
        MV_LOG_ERROR() << "Could not open configuration file:" << config_csv_path_;
        return;
    }

    for (const auto &param : init_params_) {
        switch (param.value.which()) {
        case 0:
            myfile << boost::get<unsigned int>(param.value);
            break;
        case 1:
            myfile << boost::get<float>(param.value);
            break;
        }

        myfile << kCsvDelimiter << " \t" << param.desc << std::endl;
    }

    myfile.close();
    MV_LOG_INFO() << "Configuration file has been initialized:" << config_csv_path_;
}

template<typename T>
void PsmConfigCsvLoader::write_param(PsmConfigCsvLoader::CSVParameter &p, const std::string &name, const T &min,
                                     const T &value, const T &max, const std::string &additional_info) {
    assert(min <= value && value <= max);

    std::ostringstream ss;
    ss << name << " // " << details::type_to_str<T>() << " value in range [" << min << "; " << max << "] (preferably) "
       << additional_info;

    p.desc  = ss.str();
    p.value = CSVParameter::Value(value);
}

} // namespace Metavision
