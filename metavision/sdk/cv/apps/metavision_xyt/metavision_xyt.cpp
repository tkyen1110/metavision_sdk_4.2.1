/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#include "Polyfill/OgreImGuiInputListener.h"
#include "Polyfill/OgreImGuiOverlay.h"

#include <Ogre.h>
#include <OgreApplicationContext.h>
#include <OgreCameraMan.h>
#include <OgreFileSystemLayer.h>
#include <OgreHardwareBuffer.h>
#include <OgreHardwareVertexBuffer.h>
#include <OgreHighLevelGpuProgramManager.h>
#include <OgreLog.h>
#include <OgreOverlayManager.h>
#include <OgreOverlaySystem.h>
#include <OgreResourceManager.h>
#include <OgreVector.h>

#include <imgui.h>

#include <metavision/sdk/core/utils/colors.h>
#include <metavision/sdk/cv/algorithms/anti_flicker_algorithm.h>
#include <metavision/sdk/cv/algorithms/spatio_temporal_contrast_algorithm.h>
#include <metavision/sdk/cv/algorithms/trail_filter_algorithm.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/driver/geometry.h>

#include <boost/program_options.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/variant.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace po = boost::program_options;
using Ogre::Polyfill::ImGuiOverlay;
using OgreBites::Polyfill::ImGuiInputListener;

using EventBuffer = std::vector<Metavision::EventCD>;

class CmdLineOptions {
    po::options_description options_desc_;
    po::variables_map vm_;

public:
    std::string serial;
    std::string input_file_path;
    std::string biases_file;
    bool help           = false;
    bool show_frames    = false;
    size_t num_vertices = 0;

    CmdLineOptions(int argc, char *argv[]) : options_desc_("Metavision XYT - Usage") {
        // clang-format off
        options_desc_.add_options()
        ("help,h", "Produce help message.")
        ("serial,s",          po::value<std::string>(&serial),"Serial ID of the camera. This flag is incompatible with flag '--input-file'.")
        ("input-file,i",      po::value<std::string>(&input_file_path), "Path to input file. If not specified, the camera live stream is used.")
        ("show-frames,f",     po::bool_switch(&show_frames), "Show frames from startup")
        ("biases,b",          po::value<std::string>(&biases_file), "Path to a biases file. If not specified, the camera will be configured with the default biases.")
        ("num_vertices,n",    po::value<size_t>(&num_vertices)->default_value(5'000'000), "Number of Vertices allocated in the 3d Point cloud.")
        ;
        // clang-format on

        po::store(po::command_line_parser(argc, argv).options(options_desc_).run(), vm_);
        po::notify(vm_);

        help = vm_.count("help");
        std::string in_file_path;
        if (serial.size() && input_file_path.size()) {
            throw std::invalid_argument("Error : `--serial` and `--input-file` options are exclusives.");
        }
    }

    void print_help(std::ostream &o = std::cout) {
        o << options_desc_;
    }
};

namespace {
#if OGRE_PLATFORM == OGRE_PLATFORM_LINUX
#include <X11/Xresource.h>
/*
 * When using Fractional Scaling feature on Linux, we need to get the Dot Per Inch (dpi) scale for the current display.
 * This is an implementation for Xlib implementation over X11
 */
float getPlatformDpiScale() {
    auto dpy   = XOpenDisplay(nullptr);
    auto xdefs = XResourceManagerString(dpy);

    XrmValue value;
    char *type;
    XrmDatabase xrdb = XrmGetStringDatabase(xdefs);
    Bool found       = XrmGetResource(xrdb, "Xft.dpi", "Xft.fpi", &type, &value);

    float dpi_scale = 1.f;
    if (found) {
        std::string dpi(value.addr, value.size);
        dpi_scale = std::stof(dpi) / 96.f; // Default DPI on Linux platform is 96
    }

    XrmDestroyDatabase(xrdb);
    XCloseDisplay(dpy);
    return dpi_scale;
}
#else
float getPlatformDpiScale() {
    return 1.;
}
#endif
} // namespace

enum class CameraView {
    origin,
    top,
    front,
    side,
    free,
    update_camera,
};

struct LiveConfig {
    bool quit                        = false;
    bool playing                     = false;
    float scale                      = 0.01f;
    float point_size                 = 1.f;
    int time_ms_window               = 1'000;
    bool show_frames                 = false;
    int frames_ms_time_slice         = 100;
    int frames_ms_exposure           = 10;
    Metavision::ColorPalette palette = Metavision::ColorPalette::Light;
    CameraView view                  = CameraView::origin;
    bool demo                        = false;
    float slowdown                   = 1.f;

    struct FilterConf {
        enum struct Types {
            none,
            stc,
            trail,
            afk,
        };
        struct STCConf {
            int time_threshold = 20'000;
            bool cut_trail     = true;
        } stc;
        struct TrailConf {
            int time_threshold = 20'000;
        } trail;
        struct AFKConf {
            int filter_length            = 7;
            float min_freq               = 10.f;
            float max_freq               = 150.f;
            int diff_thresh_us           = 1500;
            bool output_all_burst_events = false;
        } afk;
        Types type = Types::stc;
    } noise_filter;

    LiveConfig(const CmdLineOptions &opt) : show_frames(opt.show_frames) {
        if (opt.input_file_path.empty()) {
            slowdown = -1.f; // disable slowdown option with live camera
        }
    }
    Ogre::ColourValue get_ogre_colour(const Metavision::ColorType &type) {
        auto mv_colour = Metavision::get_color(palette, type);
        return Ogre::ColourValue(mv_colour.r, mv_colour.g, mv_colour.b);
    }
};

struct InputListener : public ImGuiInputListener {
    LiveConfig &conf_;
    InputListener(LiveConfig &conf) : conf_(conf) {}

    bool keyPressed(const OgreBites::KeyboardEvent &evt) override final {
        switch (evt.keysym.sym) {
        case 'p':
        case 'P':
            conf_.playing = !conf_.playing;
            break;
        case 'q':
        case 'Q':
            conf_.quit = true;
            break;
        case 'r':
        case 'R':
            conf_.view = CameraView::origin;
            break;
        case 'f':
        case 'F':
            conf_.view = CameraView::front;
            break;
        case 's':
        case 'S':
            conf_.view = CameraView::side;
            break;
        case 't':
        case 'T':
            conf_.view = CameraView::top;
            break;
        case 'd':
            if (evt.keysym.mod == 65) {
                conf_.demo = !conf_.demo;
            }
            break;
        }
        MV_LOG_DEBUG() << "Key pressed : " << evt.keysym.sym << " | mode : " << evt.keysym.mod << "\n";
        return ImGuiInputListener::keyPressed(evt);
    }
};

class GuiOverlay : public Ogre::RenderTargetListener {
    bool display_metrics_ = false;
    LiveConfig &conf_;
    float dpi_scale_ = 0.f;

public:
    GuiOverlay(LiveConfig &conf) : conf_(conf) {}

    void setup(Ogre::OverlayManager &overlay_mgr) {
        auto imgGuiOverlay = std::make_unique<ImGuiOverlay>();
        imgGuiOverlay->setZOrder(300);
        imgGuiOverlay->show();
        overlay_mgr.addOverlay(imgGuiOverlay.release());
        ImGui::GetIO().FontAllowUserScaling = true;
    }

    void setScaleOverlay(float dpi_scale) {
        if (dpi_scale != dpi_scale_) {
            ImGui::GetIO().FontGlobalScale = std::round(dpi_scale);
            ImGui::GetStyle().ScaleAllSizes(dpi_scale);
            dpi_scale_ = dpi_scale;
        }
    }

    void preViewportUpdate(const Ogre::RenderTargetViewportEvent &evt) override {
        if (!evt.source->getOverlaysEnabled()) {
            return;
        }

        auto background_colour = conf_.get_ogre_colour(Metavision::ColorType::Background);
        evt.source->setBackgroundColour(background_colour);

        ImGuiOverlay::NewFrame();
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem(conf_.playing ? "Pause" : "Play", "P")) {
                    conf_.playing = !conf_.playing;
                }
                ImGui::MenuItem("Quit", "Q", &conf_.quit);
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Settings")) {
                ImGui::Text("Display");
                static std::map<Metavision::ColorPalette, std::string> palettes = {
                    {Metavision::ColorPalette::Light, "Light"},
                    {Metavision::ColorPalette::Dark, "Dark"},
                    {Metavision::ColorPalette::CoolWarm, "CoolWarm"},
                    {Metavision::ColorPalette::Gray, "Gray"},
                };
                std::string &selected_palette_name = palettes[conf_.palette];
                if (ImGui::BeginCombo("Colour Palette", selected_palette_name.c_str())) {
                    for (const auto &i_pal : palettes) {
                        if (ImGui::Selectable(i_pal.second.c_str(), i_pal.second == selected_palette_name)) {
                            conf_.palette = i_pal.first;
                        }
                    }
                    ImGui::EndCombo();
                }
                if (conf_.slowdown > 0.f) {
                    ImGui::SliderFloat("Slowdown Factor", &conf_.slowdown, 0.001f, 1.f, "%.3f");
                }
                ImGui::SliderInt("Time Window", &conf_.time_ms_window, 1, 1'000, "%u ms", ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat("Display scaling", &conf_.scale, 0.f, 1.f, "%.3f", ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat("Point Size", &conf_.point_size, 0.01f, 4.0f);

                ImGui::Separator();
                ImGui::Text("Event Stream Filter");

                static std::unordered_map<LiveConfig::FilterConf::Types, std::string> noise_filters = {
                    {LiveConfig::FilterConf::Types::none, "None"},
                    {LiveConfig::FilterConf::Types::stc, "STC"},
                    {LiveConfig::FilterConf::Types::trail, "Trail"},
                    {LiveConfig::FilterConf::Types::afk, "AFK"},
                };
                if (ImGui::BeginCombo("Noise Filter", noise_filters[conf_.noise_filter.type].c_str())) {
                    for (const auto &filter : noise_filters) {
                        if (ImGui::Selectable(filter.second.c_str(), filter.first == conf_.noise_filter.type)) {
                            conf_.noise_filter.type = filter.first;
                        }
                    }
                    ImGui::EndCombo();
                }
                switch (conf_.noise_filter.type) {
                case LiveConfig::FilterConf::Types::stc:
                    ImGui::SliderInt("Time Threshold", &conf_.noise_filter.stc.time_threshold, 100, 100'000, "%d",
                                     ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
                    ImGui::Checkbox("Cut Trail", &conf_.noise_filter.stc.cut_trail);
                    break;
                case LiveConfig::FilterConf::Types::afk:
                    ImGui::SliderInt("Filter Length", &conf_.noise_filter.afk.filter_length, 1, 100, "%d");
                    ImGui::SliderFloat("Min Frequency", &conf_.noise_filter.afk.min_freq, 1.f, 20.f, "%.2f Hz");
                    ImGui::SliderFloat("Max Frequency", &conf_.noise_filter.afk.max_freq, 20.f, 1000.f, "%.2f Hz");
                    ImGui::Checkbox("Output all burst events", &conf_.noise_filter.afk.output_all_burst_events);
                    break;
                case LiveConfig::FilterConf::Types::trail:
                    ImGui::SliderInt("Time Threshold", &conf_.noise_filter.trail.time_threshold, 100, 100000, "%d");
                    break;
                case LiveConfig::FilterConf::Types::none:
                    break;
                }
                ImGui::Separator();
                ImGui::Text("Frames");
                if (ImGui::Checkbox("Show Frames", &conf_.show_frames)) {
                    conf_.view = CameraView::update_camera;
                }
                if (ImGui::SliderInt("Frame time span", &conf_.frames_ms_time_slice, 1, 500, "%u ms",
                                     ImGuiSliderFlags_Logarithmic)) {
                    conf_.frames_ms_exposure = std::min(conf_.frames_ms_exposure, conf_.frames_ms_time_slice);
                }
                ImGui::SliderInt("Frame exposure", &conf_.frames_ms_exposure, 1, conf_.frames_ms_time_slice, "%u ms");

                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                if (ImGui::MenuItem("Reset", "R")) {
                    conf_.view = CameraView::origin;
                }
                if (ImGui::MenuItem("Front", "F")) {
                    conf_.view = CameraView::front;
                }
                if (ImGui::MenuItem("Top", "T")) {
                    conf_.view = CameraView::top;
                }
                if (ImGui::MenuItem("Side", "S")) {
                    conf_.view = CameraView::side;
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("?")) {
                ImGui::MenuItem("Metrics", nullptr, &display_metrics_);
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
        if (display_metrics_) {
            ImGui::ShowMetricsWindow(&display_metrics_);
        }
        if (conf_.demo) {
            ImGui::ShowDemoWindow();
        }
    }
};

class PointCloud {
public:
    struct Vertex {
        float x, y, z, p;
    };
    using Vertices = std::vector<Vertex>;

private:
    Ogre::MeshPtr mesh_           = nullptr;
    Ogre::SubMesh *event_submesh_ = nullptr;

    Vertices readerBuffer_, writerBuffer_;
    std::mutex write_lock_;

    size_t event_vbuff_offset_ = 0;
    const LiveConfig &conf_;
    Vertices events_vertices_;
    Ogre::AxisAlignedBox bounds_;

public:
    PointCloud(const LiveConfig &conf) : conf_(conf) {}

    Ogre::MeshPtr setup(size_t max_vertices, int width, int height) {
        mesh_          = Ogre::MeshManager::getSingleton().createManual("EventCDMesh", Ogre::RGN_DEFAULT);
        event_submesh_ = create_submesh("PointCloudMat", *mesh_, max_vertices);
        bounds_ = Ogre::AxisAlignedBox(Ogre::Vector3{0.f, 0.f, 0.f}, Ogre::Vector3{float(width), float(height), 0.f});
        mesh_->load();

        return mesh_;
    }

    Ogre::SubMesh *create_submesh(const Ogre::String &material_name, Ogre::Mesh &mesh, size_t max_vertices) {
        auto vertex_data = new Ogre::VertexData();
        auto vdecl       = vertex_data->vertexDeclaration;
        size_t offset    = 0;
        offset += vdecl->addElement(0, offset, Ogre::VET_FLOAT4, Ogre::VES_POSITION).getSize();

        auto &hw_buffer_mgr = Ogre::HardwareBufferManager::getSingleton();
        auto vbuff =
            hw_buffer_mgr.createVertexBuffer(offset, max_vertices, Ogre::HardwareBufferUsage::HBU_CPU_TO_GPU, true);

        vertex_data->vertexStart = 0;
        vertex_data->vertexCount = 0;
        vertex_data->vertexBufferBinding->setBinding(0, vbuff);

        const std::string default_vs_source = R"(
            #version 330 core
            layout(location = 0) in vec4 vertex; // {x, y, timestamp, polarity }

            uniform float point_size;
            uniform vec4 negative_event_color;
            uniform vec4 positive_event_color;
            uniform mat4 mvp_matrix;

            uniform vec3 time_sliced; // { x :  0 or 1 for activation, y : the time point for slicing, z : slice duration }

            out vec4 vertex_color;
            out float evt_timetamp;
            out float sliced_window;
              
            void main() {
                float sliced_event = trunc(vertex.z/time_sliced.y)*time_sliced.y;

                gl_Position = mvp_matrix * vec4(vertex.xy, sliced_event, 1.0);
                gl_PointSize = point_size;

                evt_timetamp = vertex.z;
                sliced_window = sliced_event - time_sliced.z;
                vertex_color = mix(negative_event_color, positive_event_color, vertex.wwww);
            }
        )";
        const std::string default_ps_source = R"(
            #version 330 core
            in vec4 vertex_color;
            in float evt_timetamp;
            in float sliced_window;

            uniform float timestamp_max;

            void main(void) {
                if(evt_timetamp > timestamp_max || sliced_window > evt_timetamp ) {
                    discard;
                }

                gl_FragColor = vertex_color;
            }
        )";
        auto &program_mgr                   = Ogre::HighLevelGpuProgramManager::getSingleton();
        auto vs_prog                        = program_mgr.createProgram("point_cloud_vs", Ogre::RGN_DEFAULT, "glsl",
                                                 Ogre::GpuProgramType::GPT_VERTEX_PROGRAM);
        vs_prog->setSource(default_vs_source);
        vs_prog->load();

        auto params = vs_prog->getDefaultParameters();
        params->setNamedAutoConstant("mvp_matrix", Ogre::GpuProgramParameters::ACT_WORLDVIEWPROJ_MATRIX);
        float default_time_sliced[] = {0.f, 1.f};
        params->setNamedConstant("time_sliced", default_time_sliced, 1, 2);
        params->addSharedParameters("Shared_parameters");

        auto ps_prog = program_mgr.createProgram("point_cloud_ps", Ogre::RGN_DEFAULT, "glsl",
                                                 Ogre::GpuProgramType::GPT_FRAGMENT_PROGRAM);
        ps_prog->setSource(default_ps_source);
        ps_prog->getDefaultParameters()->addSharedParameters("Shared_parameters");
        ps_prog->load();

        auto &material_mgr         = Ogre::MaterialManager::getSingleton();
        Ogre::MaterialPtr material = material_mgr.create(material_name, Ogre::RGN_DEFAULT);
        auto pass                  = material->getTechnique(0)->getPass(0);
        pass->setVertexProgram("point_cloud_vs");
        pass->setFragmentProgram("point_cloud_ps");

        Ogre::SubMesh *sub_mesh     = mesh.createSubMesh(material_name);
        sub_mesh->useSharedVertices = false;
        sub_mesh->vertexData        = vertex_data;
        sub_mesh->setMaterial(material);
        sub_mesh->operationType = Ogre::RenderOperation::OT_POINT_LIST;

        return sub_mesh;
    };

    void reset_data() {
        std::lock_guard<std::mutex> lock(write_lock_);
        event_submesh_->vertexData->vertexStart = 0;
        event_submesh_->vertexData->vertexCount = 0;
        event_vbuff_offset_                     = 0;
        writerBuffer_.clear();
        readerBuffer_.clear();
        mesh_->_setBounds(Ogre::AxisAlignedBox{});
    }

    void push_events(const Metavision::EventCD *beg, const Metavision::EventCD *end) {
        std::lock_guard<std::mutex> lock(write_lock_);

        auto eventcd_to_vertex = [&](const Metavision::EventCD &e) {
            return PointCloud::Vertex{float(e.x), float(e.y), -float(e.t), float(e.p)};
        };
        std::transform(beg, end, std::back_inserter(writerBuffer_), eventcd_to_vertex);
    }

    void swap_buffer() {
        std::lock_guard<std::mutex> lock(write_lock_);
        std::swap(readerBuffer_, writerBuffer_);
    }

    void update_vertex_data() {
        swap_buffer();

        if (readerBuffer_.size()) {
            auto min    = readerBuffer_.back().z;
            auto max    = min + conf_.time_ms_window * 1000;
            auto scaler = (max - min) * Ogre::MeshManager::getSingleton().getBoundsPaddingFactor();

            bounds_.getMinimum().z = min + scaler;
            bounds_.getMaximum().z = max - scaler;
            mesh_->_setBounds(bounds_);
        }

        send_vertices_to_submesh(readerBuffer_, event_submesh_, event_vbuff_offset_);
        readerBuffer_.clear();
    }

    static void send_vertices_to_submesh(const Vertices &vertices, Ogre::SubMesh *submesh, size_t &vertex_buff_offset) {
        auto vertex_buff = submesh->vertexData->vertexBufferBinding->getBuffer(0);

        const size_t vbuff_available_vertices = vertex_buff->getNumVertices() - vertex_buff_offset;
        const size_t vert_to_copy             = std::min(vbuff_available_vertices, vertices.size());
        hw_buffer_copy_vertex(*vertex_buff, vertices.data(), vertex_buff_offset, vert_to_copy);

        const size_t vert_left_to_copy =
            std::min(vertices.size() - vert_to_copy, vertex_buff->getNumVertices() - vert_to_copy);
        hw_buffer_copy_vertex(*vertex_buff, vertices.data() + vert_to_copy, 0, vert_left_to_copy);

        const size_t vert_written = vert_to_copy + vert_left_to_copy;
        vertex_buff_offset += vert_written;
        vertex_buff_offset %= vertex_buff->getNumVertices();

        if (submesh->vertexData->vertexCount < vertex_buff->getNumVertices()) {
            submesh->vertexData->vertexCount += vert_written;
            submesh->vertexData->vertexCount =
                std::min(submesh->vertexData->vertexCount, vertex_buff->getNumVertices());
        }
    }

    static size_t hw_buffer_copy_vertex(Ogre::HardwareVertexBuffer &hvb, const void *src, const size_t dst_offset,
                                        const size_t cpy_len) {
        const size_t vertex_size = hvb.getVertexSize();
        const size_t offset      = dst_offset * vertex_size;
        const size_t length      = cpy_len * vertex_size;

        auto hw_buff = hvb.lock(offset, length, Ogre::HardwareBuffer::LockOptions::HBL_WRITE_ONLY);
        std::memcpy(hw_buff, src, length);
        hvb.unlock();

        return length;
    }
};

class BoundingBox {
public:
    Ogre::MeshPtr mesh_     = nullptr;
    Ogre::SubMesh *submesh_ = nullptr;

    Ogre::MeshPtr setup(int width, int height) {
        mesh_    = Ogre::MeshManager::getSingleton().createManual("PointCloudBB", Ogre::RGN_DEFAULT);
        submesh_ = create_bbx_submesh(width, height, "PointCloudBBMat", *mesh_);
        mesh_->load();
        mesh_->_updateBoundsFromVertexBuffers();
        return mesh_;
    }

private:
    Ogre::SubMesh *create_bbx_submesh(int width, int height, const Ogre::String &material_name, Ogre::Mesh &mesh) {
        auto vertex_data = new Ogre::VertexData();
        auto vdecl       = vertex_data->vertexDeclaration;
        size_t offset    = 0;
        offset += vdecl->addElement(0, offset, Ogre::VET_FLOAT3, Ogre::VES_POSITION).getSize();

        auto &hw_buffer_mgr = Ogre::HardwareBufferManager::getSingleton();

        float w = width / 2., h = height / 2., z = -1.f;
        const std::array<std::array<float, 3>, 8> vertices{
            // front face
            -w, -h, 0.f, // 0
            -w, h, 0.f,  // 1
            w, h, 0.f,   // 2
            w, -h, 0.f,  // 3
            // back face
            -w, -h, z, // 4
            -w, h, z,  // 5
            w, h, z,   // 6
            w, -h, z,  // 7
        };
        auto vbuff = hw_buffer_mgr.createVertexBuffer(offset, vertices.size(), Ogre::HardwareBufferUsage::HBU_GPU_ONLY);
        vbuff->writeData(0, vbuff->getSizeInBytes(), vertices.data(), true);

        const std::array<std::int16_t, 24> indexes{
            0, 1, 1, 2, 2, 3, 3, 0, 0, 4, 1, 5, 2, 6, 3, 7, 4, 5, 5, 6, 6, 7, 7, 4,
        };
        auto ibuff = hw_buffer_mgr.createIndexBuffer(Ogre::HardwareIndexBuffer::IT_16BIT, indexes.size(), false);
        ibuff->writeData(0, ibuff->getSizeInBytes(), indexes.data(), true);

        auto &material_mgr    = Ogre::MaterialManager::getSingleton();
        Ogre::MaterialPtr mat = material_mgr.create(material_name, Ogre::RGN_DEFAULT);
        mat->setAmbient(Ogre::ColourValue(0.9f, 0.2f, 0.0f, 0.8f));

        Ogre::SubMesh *sub_mesh           = mesh.createSubMesh("PointCloudBB");
        sub_mesh->useSharedVertices       = false;
        sub_mesh->vertexData              = vertex_data;
        sub_mesh->vertexData->vertexCount = vertices.size();
        sub_mesh->vertexData->vertexBufferBinding->setBinding(0, vbuff);
        sub_mesh->indexData->indexBuffer = ibuff;
        sub_mesh->indexData->indexCount  = indexes.size();
        sub_mesh->setMaterial(mat);
        sub_mesh->operationType = Ogre::RenderOperation::OT_LINE_LIST;

        return sub_mesh;
    }
};

class EventStreamFilter {
    using STC   = Metavision::SpatioTemporalContrastAlgorithm;
    using AFK   = Metavision::AntiFlickerAlgorithm;
    using Trail = Metavision::TrailFilterAlgorithm;

    using NoiseFilterAlgo =
        boost::variant<std::unique_ptr<STC>, std::unique_ptr<AFK>, std::unique_ptr<Trail>, std::nullptr_t>;
    NoiseFilterAlgo filter_algorithm_;

    int width_  = 0;
    int height_ = 0;
    const LiveConfig &conf_;
    LiveConfig::FilterConf::Types filter_algo_type_;

    template<class Inserter>
    struct FilterAlgoVisitor : public boost::static_visitor<> {
        const LiveConfig &conf_;
        const Metavision::EventCD *beg_;
        const Metavision::EventCD *end_;
        Inserter &inserter_;

        FilterAlgoVisitor(const LiveConfig &conf, const Metavision::EventCD *beg, const Metavision::EventCD *end,
                          Inserter &inserter) :
            conf_(conf), beg_(beg), end_(end), inserter_(inserter) {}

        void operator()(std::unique_ptr<STC> &algo) {
            algo->set_threshold(conf_.noise_filter.stc.time_threshold);
            algo->set_cut_trail(conf_.noise_filter.stc.cut_trail);
            algo->process_events(beg_, end_, inserter_);
        }
        void operator()(std::unique_ptr<AFK> &algo) {
            algo->set_filter_length(conf_.noise_filter.afk.filter_length);
            algo->set_min_freq(conf_.noise_filter.afk.min_freq);
            algo->set_max_freq(conf_.noise_filter.afk.max_freq);
            algo->set_difference_threshold(conf_.noise_filter.afk.diff_thresh_us);
            algo->process_events(beg_, end_, inserter_);
        }
        void operator()(std::unique_ptr<Trail> &algo) {
            algo->set_threshold(conf_.noise_filter.trail.time_threshold);
            algo->process_events(beg_, end_, inserter_);
        }
        void operator()(std::nullptr_t &) {
            std::copy(beg_, end_, inserter_);
        }
    };

    NoiseFilterAlgo make_noise_filter(LiveConfig::FilterConf::Types noise_filter_type) {
        switch (conf_.noise_filter.type) {
        case LiveConfig::FilterConf::Types::stc:
            return std::make_unique<STC>(width_, height_, conf_.noise_filter.stc.time_threshold);
        case LiveConfig::FilterConf::Types::afk:
            return std::make_unique<AFK>(width_, height_,
                                         Metavision::FrequencyEstimationConfig(
                                             conf_.noise_filter.afk.filter_length, conf_.noise_filter.afk.min_freq,
                                             conf_.noise_filter.afk.max_freq, conf_.noise_filter.afk.diff_thresh_us,
                                             conf_.noise_filter.afk.output_all_burst_events));
        case LiveConfig::FilterConf::Types::trail:
            return std::make_unique<Trail>(width_, height_, conf_.noise_filter.trail.time_threshold);
        case LiveConfig::FilterConf::Types::none:
        default:
            return nullptr;
        }
    }

public:
    EventStreamFilter(const LiveConfig &conf, const Metavision::Geometry &geom) :
        conf_(conf), width_(geom.width()), height_(geom.height()) {
        filter_algo_type_ = conf_.noise_filter.type;
        filter_algorithm_ = make_noise_filter(conf_.noise_filter.type);
    }

    template<class Inserter>
    void filter(const Metavision::EventCD *beg, const Metavision::EventCD *end, Inserter &&inserter) {
        if (conf_.noise_filter.type != filter_algo_type_) {
            filter_algo_type_ = conf_.noise_filter.type;
            filter_algorithm_ = make_noise_filter(filter_algo_type_);
        }
        FilterAlgoVisitor<Inserter> algo_visitor(conf_, beg, end, inserter);
        boost::apply_visitor(algo_visitor, filter_algorithm_);
    }
};

class MvCamera {
    const CmdLineOptions &opts_;
    LiveConfig &conf_;
    PointCloud point_cloud_;
    BoundingBox point_cloud_bbx_;
    std::unique_ptr<Metavision::Camera> camera_;
    bool camera_has_stopped = false;
    EventBuffer filtered_evt_;
    EventStreamFilter event_stream_filter_;

public:
    using EventCD_CB = std::function<void(const Metavision::EventCD *, const Metavision::EventCD *)>;

    MvCamera(const CmdLineOptions &opts, LiveConfig &conf) :
        opts_(opts),
        conf_(conf),
        point_cloud_(conf),
        camera_(make_camera()),
        event_stream_filter_(conf_, camera_->geometry()) {
        start();
    }

    ~MvCamera() {
        camera_->stop();
    }

    std::unique_ptr<Metavision::Camera> make_camera() {
        std::unique_ptr<Metavision::Camera> camera;
        if (opts_.input_file_path.size()) {
            Metavision::FileConfigHints camera_fileconf;
            camera_fileconf.real_time_playback(true);
            camera = std::make_unique<Metavision::Camera>(
                Metavision::Camera::from_file(opts_.input_file_path, camera_fileconf));
        } else if (opts_.serial.size()) {
            camera = std::make_unique<Metavision::Camera>(Metavision::Camera::from_serial(opts_.serial));
        } else {
            camera = std::make_unique<Metavision::Camera>(Metavision::Camera::from_first_available());
            if (opts_.biases_file.size()) {
                camera->biases().set_from_file(opts_.biases_file);
            }
        }

        camera->add_runtime_error_callback([](const Metavision::CameraException &e) { throw e; });
        camera->add_status_change_callback([this](const Metavision::CameraStatus &status) {
            switch (status) {
            case Metavision::CameraStatus::STARTED:
                camera_has_stopped = false;
                break;
            case Metavision::CameraStatus::STOPPED:
                if (conf_.playing) {
                    camera_has_stopped = true;
                }
                break;
            }
        });
        camera->cd().add_callback([&](const Metavision::EventCD *evt_beg, const Metavision::EventCD *evt_end) {
            filtered_evt_.reserve(std::distance(evt_beg, evt_end));
            event_stream_filter_.filter(evt_beg, evt_end, std::back_inserter(filtered_evt_));

            evt_beg = filtered_evt_.data();
            evt_end = evt_beg + filtered_evt_.size();

            point_cloud_.push_events(evt_beg, evt_end);
            filtered_evt_.clear();

            if (0.f < conf_.slowdown && conf_.slowdown < 1.f) {
                using namespace std::chrono_literals;
                auto delay = 1ns * conf_.slowdown + 1ms * (1 - conf_.slowdown);
                std::this_thread::sleep_for(delay);
            }
        });

        return camera;
    }

    Ogre::MeshPtr setup(size_t point_cloud_size, int with, int height) {
        return point_cloud_.setup(point_cloud_size, with, height);
    }

    Ogre::MeshPtr setup_bbx(int width, int height) {
        return point_cloud_bbx_.setup(width, height);
    }

    bool end_of_file() const {
        return camera_has_stopped && opts_.input_file_path.size();
    }

    void start() {
        camera_->start();
        conf_.playing = true;
    }

    void stop() {
        camera_->stop();
        conf_.playing = false;
    }

    void reset() {
        point_cloud_.reset_data();
        camera_ = make_camera();
    }

    void update_state() {
        point_cloud_.update_vertex_data();

        if (conf_.playing) {
            if (!camera_->is_running()) {
                camera_->start();
            }
        } else {
            if (camera_->is_running()) {
                camera_->stop();
            }
        }

        if (end_of_file()) {
            reset();
        }
    }

    Metavision::Geometry get_geometry() {
        return camera_->geometry();
    }
};

class App : public OgreBites::ApplicationContext {
    const CmdLineOptions &opts_;
    LiveConfig &conf_;
    std::unique_ptr<MvCamera> mv_camera_ = nullptr;
    GuiOverlay gui_;

    size_t camera_width_ = 0, camera_height_ = 0;
    std::unique_ptr<OgreBites::CameraMan> camera_mgr_;
    std::unique_ptr<ImGuiInputListener> gui_input_listener_;
    OgreBites::InputListenerChain listener_chain_;
    Ogre::SceneNode *point_cloud_node_       = nullptr;
    Ogre::SceneNode *point_cloud_bbx_node_   = nullptr;
    Ogre::Entity *point_cloud_entity_        = nullptr;
    Ogre::Entity *point_cloud_sliced_entity_ = nullptr;
    Ogre::MaterialPtr point_cloud_sliced_material_;

    struct Listener : public Ogre::LogListener {
        virtual void messageLogged(const Ogre::String &message, Ogre::LogMessageLevel lml, bool maskDebug,
                                   const Ogre::String &logName, bool &skipThisMessage) override {
            if (lml >= Ogre::LML_CRITICAL) {
                std::cerr << "Ogre Error > " << message << "\n";
            }
        }
    } log_listener_;

public:
    App(const std::string &app_name, const CmdLineOptions &opts, LiveConfig &conf) :
        OgreBites::ApplicationContext(app_name),
        opts_(opts),
        conf_(conf),
        mv_camera_(std::make_unique<MvCamera>(opts, conf)),
        gui_(conf),
        camera_width_(mv_camera_->get_geometry().width()),
        camera_height_(mv_camera_->get_geometry().height()) {
        std::cout << "App - " << app_name << " -- w:" << camera_width_ << " h:" << camera_height_ << " (with "
                  << opts_.num_vertices << " vertices)\n";

        auto *log_mgr = Ogre::LogManager::getSingletonPtr();
        log_mgr       = new Ogre::LogManager();
        auto log      = log_mgr->createLog(mFSLayer->getWritablePath(app_name + ".log"), true, false);
        log->addListener(&log_listener_);
    }

    void cleanUp() {
        point_cloud_sliced_material_.reset();
        mv_camera_.reset();
    }

    void set_camera_position() {
        Ogre::Real dist           = std::sqrt(std::pow(camera_width_, 2) + std::pow(camera_height_, 2));
        const Ogre::Real box_size = conf_.time_ms_window * 1000 * conf_.scale;
        if (conf_.show_frames) {
            dist *= 2.f;
        }
        switch (conf_.view) {
        case CameraView::origin:
            camera_mgr_->setYawPitchDist(Ogre::Radian(Ogre::Math::PI / 7.), Ogre::Radian(Ogre::Math::PI / 9.),
                                         Ogre::Real(dist));
            camera_mgr_->setPivotOffset({0, 0, -10'000 * conf_.scale});
            break;
        case CameraView::front:
            camera_mgr_->setYawPitchDist(Ogre::Radian(), Ogre::Radian(), Ogre::Real(dist));
            break;
        case CameraView::top:
            camera_mgr_->setYawPitchDist(Ogre::Radian(), Ogre::Radian(Ogre::Math::PI / 2.), Ogre::Real(box_size));
            camera_mgr_->setPivotOffset({0, 0, -box_size / 2.f});
            break;
        case CameraView::side:
            camera_mgr_->setYawPitchDist(Ogre::Radian(Ogre::Math::PI / 2.), Ogre::Radian(), Ogre::Real(box_size));
            camera_mgr_->setPivotOffset({0, 0, -box_size / 2.f});
            break;
        case CameraView::update_camera:
            // Send empty mouse events to force camera update over new pivot
            camera_mgr_->mousePressed({OgreBites::MOUSEBUTTONDOWN, 0, 0, OgreBites::BUTTON_LEFT, 0});
            camera_mgr_->mouseMoved({0});
            camera_mgr_->mouseReleased({OgreBites::MOUSEBUTTONUP, 0, 0, OgreBites::BUTTON_LEFT, 0});
            break;
        case CameraView::free:
            break;
        }
        conf_.view = CameraView::free;
    }

    void windowResized(Ogre::RenderWindow *rw) override {
        handleRenderWindow();
    }

    void handleRenderWindow() {
        gui_.setScaleOverlay(getPlatformDpiScale());
    }

    // Our Shaders are written for OpenGL 3 only. We don't want the user to load a legacy OpenGL, or
    // any other platform specific Graphics API like DirectX, Vulkan, Metal etc...
    bool oneTimeConfig() override {
        auto root              = getRoot();
        Ogre::RenderSystem *rs = root->getRenderSystemByName("OpenGL 3+ Rendering Subsystem");
        if (!rs) {
            throw std::runtime_error("Ogre plugin 'OpenGL 3' not found on the system. \nHave you set your "
                                     "OGRE_CONFIG_DIR=<plugins.cfg> and/or your OGRE_PLUGIN_DIR envvar ?");
        }

        std::map<std::string, std::string> config = {
            //{"Debug Layer", "Off"},
            {"Display Frequency", "N/A"},
            {"FSAA", "0"},
            {"Full Screen", "No"},
            {"RTT Preferred Mode", "FBO"},
            {"Reversed Z-Buffer", "No"},
            {"Separate Shader Objects", "No"},
            {"VSync", "Yes"},
            {"VSync Interval", "1"},
            {"Video Mode", "1024 x  768"},
            {"sRGB Gamma Conversion", "No"},
        };
        for (auto &conf_it : config) {
            try {
                rs->setConfigOption(conf_it.first, conf_it.second);
            } catch (std::exception &e) {
                /// Configuration key might not exist between 2 different platforms.
                /// Ignore the one that doesn't exist
            }
        }
        root->setRenderSystem(rs);
        return true;
    }

#ifndef _WIN32
    void locateResources() override {
        OgreBites::ApplicationContextBase::loadResources();
        auto &rgm            = Ogre::ResourceGroupManager::getSingleton();
        const auto &mediaDir = getDefaultMediaDir();
        rgm.addResourceLocation(mediaDir + "/ShadowVolume", "FileSystem", Ogre::RGN_INTERNAL);
        rgm.addResourceLocation(mediaDir + "/RTShaderLib/GLSL", "FileSystem", Ogre::RGN_INTERNAL);
        rgm.addResourceLocation(mediaDir + "/RTShaderLib/materials", "FileSystem", Ogre::RGN_INTERNAL);
    }
#endif

    void setup() override {
        OgreBites::ApplicationContext::setup();

        gui_.setup(Ogre::OverlayManager::getSingleton());
        handleRenderWindow();

        auto ogre_root = getRoot();
        ogre_root->addFrameListener(this);

        auto scnMgr = ogre_root->createSceneManager();
        scnMgr->setAmbientLight(Ogre::ColourValue::White);
        scnMgr->addRenderQueueListener(mOverlaySystem);

        Ogre::RTShader::ShaderGenerator::getSingleton().addSceneManager(scnMgr);

        auto shared_params = Ogre::GpuProgramManager::getSingleton().createSharedParameters("Shared_parameters");
        shared_params->addConstantDefinition("negative_event_color", Ogre::GpuConstantType::GCT_FLOAT4);
        shared_params->addConstantDefinition("positive_event_color", Ogre::GpuConstantType::GCT_FLOAT4);
        shared_params->addConstantDefinition("point_size", Ogre::GpuConstantType::GCT_FLOAT1);
        shared_params->addConstantDefinition("timestamp_max", Ogre::GpuConstantType::GCT_FLOAT1);

        auto cam = scnMgr->createCamera("MainCamera");
        cam->setNearClipDistance(1);
        cam->setFarClipDistance(1'000'000);
        cam->setAutoAspectRatio(true);

        Ogre::SceneNode *camera_node = scnMgr->getRootSceneNode()->createChildSceneNode();
        camera_node->attachObject(cam);

        auto window = getRenderWindow();
        window->addListener(&gui_);
        window->addViewport(cam);

        auto main_node = scnMgr->getRootSceneNode()->createChildSceneNode();

        auto point_cloud_mesh = mv_camera_->setup(opts_.num_vertices, camera_width_, camera_height_);
        point_cloud_entity_   = scnMgr->createEntity(point_cloud_mesh);

        point_cloud_node_ = main_node->createChildSceneNode();
        point_cloud_node_->attachObject(point_cloud_entity_);
        point_cloud_node_->rotate(Ogre::Vector3{0, 0, 1}, Ogre::Radian(Ogre::Math::PI));

        auto point_cloud_bbx_mesh    = mv_camera_->setup_bbx(camera_width_, camera_height_);
        auto point_cloud_bbx_entity_ = scnMgr->createEntity(point_cloud_bbx_mesh);

        point_cloud_bbx_node_ = main_node->createChildSceneNode();
        point_cloud_bbx_node_->attachObject(point_cloud_bbx_entity_);

        point_cloud_sliced_material_ =
            Ogre::MaterialManager::getSingleton().getByName("PointCloudMat")->clone("PointCloudSlicedMat");
        point_cloud_sliced_entity_ = scnMgr->createEntity(point_cloud_mesh);
        point_cloud_sliced_entity_->setMaterialName(point_cloud_sliced_material_->getName());

        auto point_cloud_sliced_node = point_cloud_node_->createChildSceneNode(Ogre::Vector3(camera_width_ * -1.f, 0.f, 0.f));
        point_cloud_sliced_node->attachObject(point_cloud_sliced_entity_);

        camera_mgr_ = std::make_unique<OgreBites::CameraMan>(camera_node);
        camera_mgr_->setStyle(OgreBites::CS_ORBIT);
        camera_mgr_->setTarget(point_cloud_bbx_node_);

        gui_input_listener_ = std::make_unique<InputListener>(conf_);
        listener_chain_     = OgreBites::InputListenerChain({gui_input_listener_.get(), camera_mgr_.get()});
        addInputListener(&listener_chain_);
    }

    void update_bbx_node() {
        Ogre::Vector3 bbx_scale(1.f, 1.f, conf_.scale * conf_.time_ms_window * 1000);
        Ogre::Vector3 bbx_pos(0.f);
        if (conf_.show_frames) {
            bbx_scale.x *= 2;
            bbx_pos.x += camera_width_ / 2.f;
        }
        point_cloud_bbx_node_->setScale(bbx_scale);
        point_cloud_bbx_node_->setPosition(bbx_pos);
    }

    bool frameRenderingQueued(const Ogre::FrameEvent &evt) override {
        mv_camera_->update_state();

        Ogre::Vector3 front_bound = point_cloud_entity_->getBoundingBox().getMinimum();
        point_cloud_node_->setPosition({camera_width_ / 2.f, camera_height_ / 2.f, front_bound.z * conf_.scale});
        point_cloud_node_->setScale(1., 1., -conf_.scale);

        update_bbx_node();

        auto &program_mgr  = Ogre::GpuProgramManager::getSingleton();
        auto shared_params = program_mgr.getSharedParameters("Shared_parameters");

        auto negative_color = conf_.get_ogre_colour(Metavision::ColorType::Negative);
        auto positive_color = conf_.get_ogre_colour(Metavision::ColorType::Positive);

        shared_params->setNamedConstant("negative_event_color", negative_color);
        shared_params->setNamedConstant("positive_event_color", positive_color);
        shared_params->setNamedConstant("point_size", conf_.point_size);
        shared_params->setNamedConstant("timestamp_max", front_bound.z + conf_.time_ms_window * 1000);

        point_cloud_sliced_entity_->setVisible(conf_.show_frames);

        point_cloud_sliced_material_->getTechnique(0)->getPass(0)->getVertexProgramParameters()->setNamedConstant(
            "time_sliced", Ogre::Vector3(1.f, conf_.frames_ms_time_slice * 1000.f, conf_.frames_ms_exposure * 1000.f));

        set_camera_position();
        return conf_.quit ? false : true;
    }

    ~App() {
        auto overlay_mgr = Ogre::OverlayManager::getSingletonPtr();
        if (overlay_mgr) {
            overlay_mgr->destroy("ImGuiOverlay");
        }

        if (mRoot) {
            closeApp();
        }
    }
};

int main(int argc, char *argv[]) {
    CmdLineOptions opts(argc, argv);
    LiveConfig conf(opts);

    if (opts.help) {
        opts.print_help();
        return 0;
    }

    try {
        App app("Metavision-xyt", opts, conf);
        try {
            app.initApp();
            app.getRoot()->startRendering();
        } catch (std::exception &e) {
            MV_LOG_ERROR() << "Runtime exception : " << e.what() << "\n";
            return 2;
        }

        app.cleanUp();

    } catch (std::exception &e) {
        MV_LOG_ERROR() << "Init exception : " << e.what() << "\n";
        return 1;
    }

    return 0;
}
