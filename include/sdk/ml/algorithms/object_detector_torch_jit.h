/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ML_ALGORITHMS_OBJECT_DETECTOR_TORCH_JIT_H
#define METAVISION_SDK_ML_ALGORITHMS_OBJECT_DETECTOR_TORCH_JIT_H

#include <string>
#include <vector>
#include <map>
#include <boost/filesystem.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/ml/algorithms/cd_processing_algorithm.h"
#include "metavision/sdk/ml/algorithms/non_maximum_suppression.h"
#include "metavision/sdk/ml/algorithms/detail/cd_processing_diff.h"
#include "metavision/sdk/ml/algorithms/detail/cd_processing_histo.h"
#include "metavision/sdk/ml/algorithms/detail/cd_processing_event_cube.h"

namespace bfs = boost::filesystem;

namespace {

/// @brief Gets a vector of elements from ptree. If the key is missing, the function will throw a std::logic_error
/// @param pt Ptree to query
/// @param key Element to retrieve
/// @param filename Filename used only to display an error message
template<typename T>
inline std::vector<T> as_vector(boost::property_tree::ptree const &pt, boost::property_tree::ptree::key_type const &key,
                                const std::string &filename) {
    std::vector<T> r;
    try {
        for (auto &item : pt.get_child(key))
            r.push_back(item.second.get_value<T>());
    } catch (const boost::property_tree::ptree_error &e) {
        std::ostringstream oss;
        oss << "Error: could not find the key '" << key << "' in file " << filename << std::endl;
        oss << e.what() << std::endl;
        throw std::logic_error(oss.str());
    }
    return r;
}

// @brief Gets an element from ptree. If the key is missing, the function will throw a std::logic_error
// @param pt Ptree to query
// @param key Element to retrieve
// @param filename Filename used only in the error message in case the key has not been found in the ptree
template<typename T>
T get_element_from_ptree(const boost::property_tree::ptree &pt, const std::string &key, const std::string &filename) {
    try {
        T value = pt.get<T>(key);
        return value;
    } catch (const boost::property_tree::ptree_error &e) {
        std::ostringstream oss;
        oss << "Error: could not find the key '" << key << "' in file " << filename << std::endl;
        oss << e.what() << std::endl;
        throw std::logic_error(oss.str());
    }
}

/// @brief Parses a JSON file
/// @param filename Json file
/// @param required Check if the file does not exist and raised a exception
/// @return tree Tree describing the json structure
/// @throw runtime_error if the file does not exist and required is equal to true
static boost::property_tree::ptree get_tree_from_file(const std::string &filename, bool required = true) {
    boost::property_tree::ptree pt;

    std::stringstream file_buffer;
    std::ifstream file(filename);

    if (file) {
        file_buffer << file.rdbuf();
        file.close();
    } else if (required) {
        throw std::runtime_error(std::string(" No such file: '") + filename + "'");
    }

    try {
        boost::property_tree::read_json(file_buffer, pt);
    } catch (std::exception &e) {
        throw std::runtime_error(e.what() + std::string(" reading '") + filename + "'");
    } catch (...) { throw std::runtime_error("Unknown exception thrown reading '" + filename + "'"); }
    return pt;
}

/// @brief Gets the ignore classes from file
/// @param filename File containing class ignored stored into a JSON file
/// @param num_classes Number of classes contained into the network
/// @param ignored_classes Vector to store classes that will be ignored
static void parse_json_ignore_classes(const std::string &filename, const int num_classes,
                                      std::vector<int> &ignored_classes) {
    boost::property_tree::ptree pt = get_tree_from_file(filename);

    try {
        ignored_classes = as_vector<int>(pt, "ignore_classes", filename);
    } catch (const std::logic_error &e) {
        MV_SDK_LOG_TRACE() << filename << "has no ignored_classes. Using all classes.";
        ignored_classes.clear();
    }

    for (auto it = ignored_classes.begin(); it != ignored_classes.end(); ++it) {
        if (*it >= num_classes) {
            std::ostringstream oss;
            oss << "ERROR: invalid ignored class index: " << *it << ". The network has " << num_classes
                << " outputs (including 'background')" << std::endl;
            throw std::invalid_argument(oss.str());
        }
    }
}

/// @brief Parses JSON file and set the output classes labels
/// @param filename Json filename which should contain the key "label_map"
/// @param num_classes Number of classes contained into the network
/// @param labels Name of the ignored classes
static void parse_json_label_map(const std::string &filename, const int num_classes, std::vector<std::string> &labels) {
    boost::property_tree::ptree pt = get_tree_from_file(filename);

    try {
        labels = as_vector<std::string>(pt, "label_map", filename);
    } catch (const std::logic_error &e) {
        MV_SDK_LOG_WARNING() << filename
                             << "has no label_map. Please update the file.\n"
                                "In a not so distant future, this will trigger an error rather than a warning.";
        labels.clear();
        assert(num_classes > 1);
        labels.reserve(num_classes);
        labels.push_back("background");
        for (int i = 1; i < num_classes; ++i) {
            std::ostringstream oss;
            oss << i;
            labels.push_back(oss.str());
        }
    }

    if (static_cast<int>(labels.size()) != num_classes) {
        std::ostringstream oss;
        oss << "Error: mismatch between the number of classes (" << num_classes << ") and the number of labels (";
        oss << labels.size() << ")  in file: " << filename << std::endl;
        throw std::logic_error(oss.str());
    }

    if ((labels.size() < 2) || (labels[0] != "background")) {
        std::ostringstream oss;
        oss << "Error: something is wrong with the label_map in file: " << filename << std::endl;
        oss << "First class MUST be 'background', and there should be at least a second class" << std::endl;
        throw std::logic_error(oss.str());
    }
}

/// @brief Gets information about the frame required by the network
/// @param filename Json file name
/// @param cd_transfo_type Name of the frame generation method
/// @param accumulation_time Size of event batch used to generate a frame
/// @param num_channels Number of channels in the generated frame
/// @param max_incr_per_pixel Maximum number of increments (events) per pixel at full resolution
/// @param clip_value_after_normalization Clipping value after normalization (typically 1.)
static void parse_json_cd_processing(const std::string &filename, std::string &cd_transfo_type,
                                     Metavision::timestamp &accumulation_time, int &num_channels,
                                     float &max_incr_per_pixel, float &clip_value_after_normalization) {
    boost::property_tree::ptree pt = get_tree_from_file(filename);

    cd_transfo_type    = get_element_from_ptree<std::string>(pt, "preprocessing_name", filename);
    accumulation_time  = get_element_from_ptree<Metavision::timestamp>(pt, "delta_t", filename);
    num_channels       = get_element_from_ptree<int>(pt, "num_channels", filename);
    max_incr_per_pixel = get_element_from_ptree<float>(pt, "max_incr_per_pixel", filename);
    if (max_incr_per_pixel <= 0.f) {
        throw std::runtime_error("max_incr_per_pixel must be strictly greater than 0");
    }
    clip_value_after_normalization = get_element_from_ptree<float>(pt, "clip_value_after_normalization", filename);
    if (clip_value_after_normalization < 0.f) {
        throw std::runtime_error("clip_value_after_normalization must be equal or greater than zero");
    }
}

/// @brief Parse json to retrieve specific entries related to event_cube preprocessing
/// @param filename Json file name
/// @param num_utbins Number of micro temporal bins
/// @param split_polarity if true, positive and negative events are processed in separate channels
static void parse_json_cd_processing_event_cube(const std::string filename, int &num_utbins, bool &split_polarity) {
    std::string cd_transfo_type             = "";
    Metavision::timestamp accumulation_time = 0;
    int num_channels                        = 0;
    float max_incr_per_pixel                = 0.f;
    float clip_value_after_normalization    = 0.f;
    parse_json_cd_processing(filename, cd_transfo_type, accumulation_time, num_channels, max_incr_per_pixel,
                             clip_value_after_normalization);

    assert(cd_transfo_type == "event_cube");
    boost::property_tree::ptree pt = get_tree_from_file(filename);
    split_polarity                 = get_element_from_ptree<bool>(pt, "split_polarity", filename);
    num_utbins                     = get_element_from_ptree<float>(pt, "num_utbins", filename);
    if (split_polarity) {
        assert(num_channels == 2 * num_utbins);
    } else {
        assert(num_channels == num_utbins);
    }
}

/// @brief Gets information to decode boxes generated by the network
/// @param filename Json file name
/// @param num_classes Number of classes contained into the network
/// @param confidence_threshold Minimal confidence value to consider boxes
/// @iou_threshold Minimal ratio of intersection area or union area
/// @param num_anchor_boxes Number of anchor boxes returned by the network
static void parse_json_ssd_box_decoding(const std::string &filename, int &num_classes, float &confidence_threshold,
                                        float &iou_threshold, int &num_anchor_boxes) {
    boost::property_tree::ptree pt = get_tree_from_file(filename);

    num_classes          = get_element_from_ptree<int>(pt, "num_classes", filename);
    confidence_threshold = get_element_from_ptree<float>(pt, "confidence_threshold", filename);
    iou_threshold        = get_element_from_ptree<float>(pt, "iou_threshold", filename);
    num_anchor_boxes     = get_element_from_ptree<float>(pt, "num_anchor_boxes", filename);
}
} // namespace

namespace Metavision {

class ObjectDetectorTorchJit {
public:
    /// @brief Constructor for ObjectDetectorTorchJit
    ///
    /// @param directory Name of the directory containing at least two files:
    ///                         - model.ptjit : PyTorch model exported using torch.jit
    ///                         - info_ssd_jit.json : JSON file which contains several information about the neural
    ///                                               network (type of input features, dimensions, accumulation time,
    ///                                               list of classes, default thresholds, etc.)
    /// @param frame_width Sensor's width
    /// @param frame_height Sensor's height
    /// @param network_input_width Neural network's width which could be smaller than frame_width. In this case
    /// the network will work on a downscaled size
    /// @param network_input_height Neural network's height which could be smaller than frame_height. In this case
    /// the network will work a downscaled size
    /// @param use_cuda Boolean to indicate if we use gpu or not
    /// @param ignore_first_n_prediction_steps Number of discarded neural network predictions at the beginning of
    ///                                        a sequence. Depending on initial conditions, recurrent models sometimes
    ///                                        have a transitory regime in which they initially produce unreliable
    ///                                        detections before they enter normal working regime.
    /// @param gpu_id GPU identification number that allows the selection of the gpu if several are available.
    ///
    /// @note When network_input_width and network_input_height are different from frame_width and frame_height,
    /// the corresponding rescaling is performed on the output bounding boxes, such that the output detection
    /// are still returned in the original input frame of the events
    ObjectDetectorTorchJit(const std::string &directory, int frame_width, int frame_height, int network_input_width = 0,
                           int network_input_height = 0, bool use_cuda = false, int ignore_first_n_prediction_steps = 0,
                           int gpu_id = 0) :
        input_width_(frame_width),
        input_height_(frame_height),
        ignore_first_n_prediction_steps_(ignore_first_n_prediction_steps) {
        const std::string network_jit_filename  = (bfs::path(directory) / "model.ptjit").string();
        const std::string network_json_filename = (bfs::path(directory) / "info_ssd_jit.json").string();

        if (network_input_width == 0) {
            network_input_width_ = frame_width;
        } else {
            network_input_width_ = network_input_width;
        }

        if (network_input_height == 0) {
            network_input_height_ = frame_height;
        } else {
            network_input_height_ = network_input_height;
        }

        std::string transfo_type;
        float max_incr_per_pixel             = 0.f;
        float clip_value_after_normalization = 0.f;
        parse_json_cd_processing(network_json_filename, transfo_type, accumulation_time_, network_input_channels_,
                                 max_incr_per_pixel, clip_value_after_normalization);
        if ((transfo_type == "diff") || (transfo_type == "diff3d")) {
            assert(network_input_channels_ == 1);
            cd_processor_.reset(new CDProcessingDiff(accumulation_time_, network_input_width_, network_input_height_,
                                                     max_incr_per_pixel, clip_value_after_normalization, frame_width,
                                                     frame_height));
        } else if ((transfo_type == "histo") || (transfo_type == "histo3d")) {
            assert(network_input_channels_ == 2);
            cd_processor_.reset(new CDProcessingHisto(accumulation_time_, network_input_width_, network_input_height_,
                                                      max_incr_per_pixel, clip_value_after_normalization, frame_width,
                                                      frame_height, true));
        } else if (transfo_type == "event_cube") {
            int num_utbins      = 0;
            bool split_polarity = true;
            parse_json_cd_processing_event_cube(network_json_filename, num_utbins, split_polarity);
            cd_processor_.reset(new CDProcessingEventCube(
                accumulation_time_, network_input_width_, network_input_height_, num_utbins, split_polarity,
                max_incr_per_pixel, clip_value_after_normalization, frame_width, frame_height));
        } else {
            throw std::logic_error("Unknown cd_processing type: " + transfo_type);
        }

        int num_anchor_boxes;
        float iou_threshold = 0.f;
        parse_json_ssd_box_decoding(network_json_filename, network_num_classes_, detection_threshold_, iou_threshold,
                                    num_anchor_boxes);
        assert(num_anchor_boxes == 0);

        parse_json_label_map(network_json_filename, network_num_classes_, labels_);

        std::vector<int> ignored_classes_vect;
        parse_json_ignore_classes(network_json_filename, network_num_classes_, ignored_classes_vect);

        model_ = torch::jit::load(network_jit_filename);
        model_.get_method("reset_all")({});
        model_.eval();

        for (const auto& params : model_.parameters()) {
            is_half_ = params.dtype() == torch::kHalf;
            if (is_half_) {
                MV_SDK_LOG_INFO() << "Loaded half precision model";
            }
            break;
        }

        if (use_cuda) {
            bool gpu_available = use_gpu_if_available(gpu_id);
            if (!gpu_available) {
                MV_SDK_LOG_WARNING() << "GPU not available! Trying to run on CPU.";
                // half precision suported only in gpu 
                assert(!is_half_);
            }
        } 


        nms_with_rescaling_ =
            Metavision::NonMaximumSuppressionWithRescaling(network_num_classes_, input_width_, input_height_,
                                                           network_input_width_, network_input_height_, iou_threshold);
        nms_with_rescaling_.ignore_class_id(0); ///< class 0 is background noise
        for (auto ignored_class : ignored_classes_vect) {
            nms_with_rescaling_.ignore_class_id(ignored_class);
        }
    }

    /// @brief Performs all computations on the CPU
    void use_cpu() {
        device_ = torch::Device(torch::kCPU);
        model_.to(device_);
    }

    /// @brief Performs the computations on the GPU if there is one
    /// @param gpu_id ID of the gpu on which the computations must be performed
    /// @return Boolean to indicate if the provided gpu_id is available
    bool use_gpu_if_available(int gpu_id = 0) {
        bool gpu_is_available = false;
        device_               = torch::Device(torch::kCPU);
        if (torch::cuda::is_available()) {
            device_          = torch::Device(torch::kCUDA, gpu_id);
            gpu_is_available = true;
        }
        model_.to(device_);
        return gpu_is_available;
    }

    /// @brief Computes the detection given the provided input tensor
    /// @param input Chunk of memory which corresponds to input tensor
    /// @param bbox_first Output iterator to add the detection boxes
    /// @param ts Timestamp of current timestep. Output boxes will have this timestamp
    template<typename OutputIt>
    inline void process(Frame_t &input, OutputIt bbox_first, timestamp ts) {
        assert(ts == ts_ + get_accumulation_time()); // the process() function must be called with the right timestamp

        torch::NoGradGuard no_grad_guard;

        const int T = 1;
        const int N = 1;
        const int C = network_input_channels_;
        const int H = network_input_height_;
        const int W = network_input_width_;

        assert(static_cast<int>(input.size()) == (T * N * C * H * W));
        torch::Tensor input_tensor = torch::from_blob(input.data(), {T, N, C, H, W});

        if (is_half_) {
            input_tensor = input_tensor.to(torch::kHalf);
        } 

        auto boxesListList = model_.forward({input_tensor.to(device_), detection_threshold_}).toList();
        nb_prediction_steps_computed_++;
        ts_ = ts;

        if (!init_done_ && nb_prediction_steps_computed_ <= ignore_first_n_prediction_steps_) {
            return;
        }
        init_done_ = true;

        assert(boxesListList.size() == 1);
        auto boxesList = boxesListList.get(0).toTensorList();
        assert(boxesList.size() == 1);
        torch::Tensor boxes = boxesList[0];

        if (device_ == torch::Device(torch::kCUDA)) {
            boxes = boxes.to(torch::Device(torch::kCPU));
        }

        assert(boxes.scalar_type() == torch::kFloat32);
        assert(boxes.is_contiguous());

        assert(boxes.dim() == 2);
        assert(boxes.sizes()[1] == 6);

        bboxes_.clear();

        for (auto i = 0; i < boxes.sizes()[0]; ++i) {
            EventBbox box;
            box.t                = ts;
            box.x                = boxes[i][0].item<float>();
            box.y                = boxes[i][1].item<float>();
            box.w                = boxes[i][2].item<float>() - box.x;
            box.h                = boxes[i][3].item<float>() - box.y;
            box.class_confidence = boxes[i][4].item<float>();
            box.class_id         = static_cast<unsigned int>(boxes[i][5].item<float>());
            assert(box.class_id != 0);
            bboxes_.push_back(box);
        }

        nms_with_rescaling_.process_events(bboxes_.begin(), bboxes_.end(), bbox_first);
    }

    /// @brief Returns true if the model runs at half precision 
    /// @return Whether the model runs at half precision 
    inline bool is_half() const {
        return is_half_;
    }

    /// @brief Returns the input frame height
    /// @return Network input height in pixels
    inline int get_network_height() const {
        return network_input_height_;
    }

    /// @brief Returns the input frame width
    /// @return Network input width in pixels
    inline int get_network_width() const {
        return network_input_width_;
    }

    /// @brief Returns the number of channels in the input frame
    /// @return Network input channel number
    inline int get_network_input_channels() const {
        return network_input_channels_;
    }

    /// @brief Returns the network input size
    /// @return Size of the input frame
    inline int get_network_input_size() const {
        return network_input_channels_ * network_input_height_ * network_input_width_;
    }

    /// @brief Returns the time during which the events are accumulated to compute the NN input tensor
    /// @return Delta time used to generate the input frame
    inline Metavision::timestamp get_accumulation_time() const {
        return static_cast<Metavision::timestamp>(accumulation_time_);
    }

    /// @brief Returns the object responsible for computing the content of the input tensor
    /// @return CDProcessing to ease the input frame generation
    CDProcessing &get_cd_processor() {
        return *cd_processor_;
    };

    /// @brief Returns a vector of labels for the classes of the neural network
    /// @return Vector of strings containing labels
    const std::vector<std::string> &get_labels() const {
        return labels_;
    }

    /// @brief Initializes the internal timestamp of the object detector
    ///
    /// This is needed in order to use the start_ts parameter in the pipeline to start at a ts > 0
    ///
    /// @param ts time at which the first slice of time starts
    void set_ts(Metavision::timestamp ts) {
        ts_ = ts;
    }

    /// @brief Uses this detection threshold instead of the default value read from the JSON file
    ///
    /// This is the lower bound on the confidence score for a detection box to be accepted.
    /// It takes values in range ]0;1[
    /// Low value  -> more detections
    /// High value -> less detections
    ///
    /// @param threshold Lower bound on the detector confidence score
    void set_detection_threshold(float threshold) {
        detection_threshold_ = threshold;
    }

    /// @brief Uses this IOU threshold for NMS instead of the default value read from the JSON file
    ///
    /// Non-Maximum suppression discards detection boxes which are too similar to each other, keeping only
    /// the best one of such group. This similarity criterion is based on the measure of Intersection-Over-Union
    /// between the considered boxes.
    /// This threshold is the upper bound on the IOU for two boxes to be considered distinct (and therefore
    /// not filtered out by the Non-Maximum Suppression). It takes values in range ]0;1[
    /// Low value  -> less overlapping boxes
    /// High value -> more overlapping boxes
    ///
    /// @param threshold Upper bound on the IOU for two boxes to be considered distinct
    void set_iou_threshold(float threshold) {
        nms_with_rescaling_.set_iou_threshold(threshold);
    }

    /// @brief Resets the memory cells of the neural network
    ///
    /// Neural networks used as object detectors are usually RNNs (typically LSTMs). Use this function
    /// to reset the memory of the neural network when feeding new inputs unrelated to the previous
    /// ones : call reset() before applying the same object detector on a new sequence
    void reset() {
        model_.get_method("reset_all")({});
    }

private:
    const int input_width_, input_height_;
    int network_input_width_, network_input_height_, network_input_channels_;
    int network_num_classes_; ///< including the background class, which must be idx 0
    std::vector<std::string> labels_;
    unsigned int nb_prediction_steps_computed_ = 0;
    unsigned int ignore_first_n_prediction_steps_;
    timestamp accumulation_time_ = 0;

    std::vector<EventBbox> bboxes_;

    std::unique_ptr<CDProcessing> cd_processor_;
    torch::Device device_ = torch::Device(torch::kCPU);
    torch::jit::script::Module model_;
    float detection_threshold_;
    bool init_done_ = false;
    bool is_half_ = false;

    Metavision::timestamp ts_                 = 0;
    unsigned int nb_prediction_steps_computed = 0;

    Metavision::NonMaximumSuppressionWithRescaling nms_with_rescaling_;

    std::map<Metavision::timestamp, std::vector<EventBbox>> debug_detections_;
};

} // namespace Metavision

#endif // METAVISION_SDK_ML_ALGORITHMS_OBJECT_DETECTOR_TORCH_JIT_H
