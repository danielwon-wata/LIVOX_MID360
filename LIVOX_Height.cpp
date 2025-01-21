//
// The MIT License (MIT)
//

#include "livox_lidar_def.h"
#include "livox_lidar_api.h"
#include "zmq.hpp"

#ifdef _WIN32
#include <winsock2.h>
#else
#include <unistd.h>
#include <arpa/inet.h>
#endif

#include "json.hpp" 
#include <regex>

#include <sstream>
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <fstream>
#include <algorithm>

#include <omp.h> // 병렬처리
#include <future>
#include <atomic>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/common/centroid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <unordered_map>
#include <Eigen/Dense>












// ----------------------------------------------------------------------------
// JSON 구조체
// ----------------------------------------------------------------------------
struct StageROI {
    std::string label = "";   // "stage1"
    int roi_x_start=0;     // mm
    int roi_x_end=0;       // mm
};

struct WATAConfig {
    int iteration = 0;

    int roi_y_start=0;  // mm
    int roi_y_end=0;    // mm
    int roi_z_start=0;  // mm
    int roi_z_end=0;    // mm
    int angle=0;        // deg

    int V_x_start=0;
    int V_x_end=0;

    bool read_file=false;
    bool save_file=false;
    std::string read_file_name="";
    std::string save_file_name="";
    // 추가
    int mean_k=0;
    float threshold=0;
    int height_threshold=0;
};

struct PalletInfo {
    bool is_pallet;
    float P_height;
};

// ----------------------------------------------------------------------------
// 전역
// ----------------------------------------------------------------------------
std::mutex control_mutex;
std::mutex g_mutex;
bool V_start_process = false;
bool reading_active = true; // 초기 상태는 읽기 활성화
bool is_paused = false; // 초기 상태는 일시정지 아님

bool heightCalibration_mode = false; // 높이 캘리브레이션 
float fixed_ground_height = 0.0f;
bool ground_height_fixed = false;
bool showGroundROI = false;    // 지면 캘리브레이션용 ROI 박스를 표시할지 여부

bool READ_PCD_FROM_FILE = false;

//int iteration = 500;
int vector_size = 8;

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_loaded(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_merge(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_raw(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZ>);

std::vector<float> x_lengths(vector_size);
std::vector<float> y_lengths(vector_size);
std::vector<float> z_lengths(vector_size);

std::vector<StageROI> g_stages;
std::vector<std::string> cluster_box_ids;
std::vector<std::string> volume_line_ids; 
std::vector<std::string> pallet_line_ids;
std::vector<std::string> pallet_height_text_ids;

float angle_degrees = 0;
int x_index = 0;
int y_index = 0;
int z_index = 0;
int point_index = 0;
int pallet_line_index = 0;

float max_x_value = 0;
float result_length = 0;
float result_height = 0;
float result_width = 0;

std::ostringstream oss;

// ----------------------------------------------------------------------------
// JSON 설정 읽기
// ----------------------------------------------------------------------------
using json = nlohmann::json;

std::string removeComments(const std::string& jsonString) {
    std::regex singleLineCommentRegex("//.*?$");
    std::regex multiLineCommentRegex("/\\*.*?\\*/");

    // 단일 및 다중 행 주석 제거
    std::string withoutComments = std::regex_replace(jsonString, multiLineCommentRegex, "");
    withoutComments = std::regex_replace(withoutComments, singleLineCommentRegex, "", std::regex_constants::format_default);

    // 각 줄을 트림하여 정리
    std::stringstream ss(withoutComments);
    std::string cleanedJson, line;
    while (std::getline(ss, line)) {
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        if (!line.empty()) {
            cleanedJson += line + "\n";
        }
    }
    return cleanedJson;
}


WATAConfig readConfigFromJson(const std::string& filePath) {
    WATAConfig cfg;

    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // 주석 제거
    std::string jsonWithoutComments = removeComments(content);

    json j;
    try {
        j = json::parse(jsonWithoutComments);
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "JSON parse error at byte " << e.byte << ": " << e.what() << std::endl;
        throw std::runtime_error("JSON parse error: " + std::string(e.what()));
    }

    cfg.iteration = j.value("iteration", 0);

    if (j.contains("roi")) {
        const auto& roi = j["roi"];
        cfg.roi_y_start = roi.value("roi_y_start", 0);
        cfg.roi_y_end = roi.value("roi_y_end", 0);
        cfg.roi_z_start = roi.value("roi_z_start", 0);
        cfg.roi_z_end = roi.value("roi_z_end", 0);
        cfg.angle = roi.value("roi_angle", 0);
        cfg.V_x_start = roi.value("V_x_start", 0);
        cfg.V_x_end = roi.value("V_x_end", 0);

        if (roi.contains("heights")) {
            for (const auto& stage : roi["heights"]) {
                StageROI st;
                st.label = stage.value("label", "");
                st.roi_x_start = stage.value("roi_x_start", 0);
                st.roi_x_end = stage.value("roi_x_end", 0);
                g_stages.push_back(st);
            }
        }
    }

    if (j.contains("file")) {
        const auto& fileConfig = j["file"];
        cfg.read_file = fileConfig.value("read_file", false);
        cfg.save_file = fileConfig.value("save_file", false);
        cfg.read_file_name = fileConfig.value("read_file_name", "");
        cfg.save_file_name = fileConfig.value("save_file_name", "");
    }

    if (j.contains("removeOutlier")) {
        const auto& filterConfig = j["removeOutlier"];
        cfg.mean_k = filterConfig.value("mean_k", 0);
        cfg.threshold = filterConfig.value("threshold", 0.0f);
    }

    cfg.height_threshold = j.value("height_threshold", 0);

    return cfg;
}

/*
    bool in_heights = false;
    std::string line;
    while (std::getline(file, line)) {

        if (line.find("\"heights\":") != std::string::npos) {
            in_heights = true;
            continue;
        }
        if (in_heights && line.find("]") != std::string::npos) {
            in_heights = false;
            continue;
        }
        if (in_heights) {
            // parse stage
            StageROI st;
            // "label"
            size_t posLabel = line.find("\"label\"");
            if (posLabel != std::string::npos) {
                size_t stPos = line.find("\"", posLabel + 7);
                size_t edPos = line.find("\"", stPos + 1);
                st.label = line.substr(stPos + 1, edPos - (stPos + 1));
            }
            // "roi_x_start"
            {
                size_t pos = line.find("\"roi_x_start\"");
                if (pos != std::string::npos) {
                    size_t c = line.find(":", pos);
                    st.roi_x_start = std::stoi(line.substr(c + 1));
                }
            }
            // "roi_x_end"
            {
                size_t pos = line.find("\"roi_x_end\"");
                if (pos != std::string::npos) {
                    size_t c = line.find(":", pos);
                    st.roi_x_end = std::stoi(line.substr(c + 1));
                }
            }
            g_stages.push_back(st);
        }
        else {
            // roi_y_start
            if (line.find("\"roi_y_start\":") != std::string::npos) {
                cfg.roi_y_start = std::stoi(line.substr(line.find(":") + 1));
            }
            else if (line.find("\"iteration\":") != std::string::npos) {
                cfg.iteration = std::stoi(line.substr(line.find(":") + 1));
            }
            else if (line.find("\"roi_y_end\":") != std::string::npos) {
                cfg.roi_y_end = std::stoi(line.substr(line.find(":") + 1));
            }
            else if (line.find("\"roi_z_start\":") != std::string::npos) {
                cfg.roi_z_start = std::stoi(line.substr(line.find(":") + 1));
            }
            else if (line.find("\"roi_z_end\":") != std::string::npos) {
                cfg.roi_z_end = std::stoi(line.substr(line.find(":") + 1));
            }
            else if (line.find("\"roi_angle\":") != std::string::npos) {
                cfg.angle = std::stoi(line.substr(line.find(":") + 1));
            }
            else if (line.find("\"V_x_start\":") != std::string::npos) {
                cfg.V_x_start = std::stoi(line.substr(line.find(":") + 1));
            }
            else if (line.find("\"V_x_end\":") != std::string::npos) {
                cfg.V_x_end = std::stoi(line.substr(line.find(":") + 1));
            }
            else if (line.find("\"read_file\":") != std::string::npos) {
                std::string val = line.substr(line.find(":") + 1);
                cfg.read_file = (val.find("true") != std::string::npos);
            }
            else if (line.find("\"save_file\":") != std::string::npos) {
                std::string val = line.substr(line.find(":") + 1);
                cfg.save_file = (val.find("true") != std::string::npos);
            }
            else if (line.find("\"read_file_name\":") != std::string::npos) {
                std::string val = line.substr(line.find(":") + 2);
                val.erase(remove(val.begin(), val.end(), '\"'), val.end());
                val.erase(remove(val.begin(), val.end(), ','), val.end());
                cfg.read_file_name = val;
            }
            else if (line.find("\"save_file_name\":") != std::string::npos) {
                std::string val = line.substr(line.find(":") + 2);
                val.erase(remove(val.begin(), val.end(), '\"'), val.end());
                val.erase(remove(val.begin(), val.end(), ','), val.end());
                cfg.save_file_name = val;
            }
            // --- filter ---
            else if (line.find("\"mean_k\":") != std::string::npos) {
                cfg.mean_k = std::stoi(line.substr(line.find(":") + 1));
            }
            else if (line.find("\"threshold\":") != std::string::npos) {
                cfg.threshold = std::stof(line.substr(line.find(":") + 1));
            }
            // --- height_threshold ---
            else if (line.find("\"height_threshold\":") != std::string::npos) {
                cfg.height_threshold = std::stoi(line.substr(line.find(":") + 1));
            }
        }
    }
    file.close();
    return cfg;
}
*/

// ----------------------------------------------------------------------------
// 기타 유틸
// ----------------------------------------------------------------------------
std::string getCurrentTime() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* local_tm = std::localtime(&now_time);
    std::ostringstream oss;
    oss << std::put_time(local_tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

void saveToFile(const std::string& json_result) {
    const std::string filename = "log/result.txt";
    std::ifstream infile(filename);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(infile, line)) {
        lines.push_back(line);
    }
    infile.close();

    lines.push_back(json_result);
    if (lines.size() > 500) {
        lines.erase(lines.begin());
    }
    std::ofstream outfile(filename);
    for (const auto& l : lines) {
        outfile << l << std::endl;
    }
    outfile.close();
}

// 포인트 클라우드 데이터 초기화 함수
void resetPointCloudData() {
    std::lock_guard<std::mutex> lock(g_mutex);
    cloud_pcd->clear();
    cloud_raw->clear();
    cloud_merge->clear();
    // 다른 클라우드 데이터도 초기화 필요 시 추가
    x_lengths.clear();
    y_lengths.clear();
    z_lengths.clear();

    std::cout << "[INFO] Point cloud data has been reset due to V_start_process state change." << std::endl;
}

void updateCameraPositionText(pcl::visualization::PCLVisualizer::Ptr viewer) {
    pcl::visualization::Camera camera;
    viewer->getCameraParameters(camera);

    std::ostringstream oss;
    oss << "Camera Position: ("
        << camera.pos[0] << ", "
        << camera.pos[1] << ", "
        << camera.pos[2] << ", "
        << camera.pos[3] << ", "
        << camera.pos[4] << ", "
        << camera.pos[5] << ")";

    // 기존 텍스트 제거
    viewer->removeShape("camera_position_text");

    // 새로운 텍스트 추가
    viewer->addText(oss.str(), 20, 700, 15, 1.0, 1.0, 1.0, "camera_position_text");
}


// ----------------------------------------------------------------------------
// 리소스 콜백 (실시간 모드)
// ----------------------------------------------------------------------------
void PointCloudCallback(uint32_t handle, const uint8_t dev_type, LivoxLidarEthernetPacket* data, void* client_data) {
    if (!data) return;

    const WATAConfig* config = static_cast<const WATAConfig*>(client_data);
    if (data->data_type == kLivoxLidarCartesianCoordinateHighData) {
        auto* p_point_data = reinterpret_cast<LivoxLidarCartesianHighRawPoint*>(data->data);
        std::lock_guard<std::mutex> lock(g_mutex);

        for (uint32_t i = 0; i < data->dot_num; i++) {
            pcl::PointXYZ point;
            point.x = p_point_data[i].x / 1000.0f;
            point.y = p_point_data[i].y / 1000.0f;
            point.z = p_point_data[i].z / 1000.0f;

            cloud_raw->points.push_back(point);
        }

        cloud_raw->width = cloud_raw->points.size();
        cloud_raw->height = 1;

        // 일정 포인트(=96*iteration=9600)마다 cloud_merge에 복사
        if (cloud_raw->points.size() >= 96 * config->iteration) {
            pcl::copyPointCloud(*cloud_raw, *cloud_merge);
            cloud_raw->clear();
        }
    }
}

// ----------------------------------------------------------------------------
// 전처리 1) Voxel
// ----------------------------------------------------------------------------
void voxelizePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    float x_leaf, float y_leaf, float z_leaf)
{
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(x_leaf, y_leaf, z_leaf);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
    voxel_filter.filter(*filtered);

    cloud->clear();
    *cloud = *filtered;
}

// ----------------------------------------------------------------------------
// 전처리 2) Outlier Remove
// ----------------------------------------------------------------------------
void removeOutliers(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const WATAConfig& config) {
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);

    sor.setMeanK(config.mean_k);
    sor.setStddevMulThresh(config.threshold);  // 더 커지면 제거가 줄고, 더 작아지면 제거가 많아짐

    sor.filter(*cloud);
}

// ----------------------------------------------------------------------------
// [자동 높이 캘리브레이션] 지면 탐지
// ----------------------------------------------------------------------------
float calculateGroundHeight(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground) {
    float ground_height_sum = 0.0f;
    int ground_point_count = 0;

    for (const auto& pt : cloud_ground->points) {
        if (pt.x >= -4.0f && pt.x <= -2.3f &&
            pt.y >= -4.48f && pt.y <= -2.35f &&
            pt.z >= -0.2f && pt.z <= 1.1f)
        {
            ground_height_sum += pt.x;
            ground_point_count++;
        }
    }
    if (ground_point_count == 0) {
        throw std::runtime_error("No ground points detected in ROI => Can't calibrate ground!");
    }
    return (ground_height_sum / ground_point_count);
}

void showGroundROIBox(pcl::visualization::PCLVisualizer* viewer, bool show) {
    // ROI 상자를 표시 또는 제거
    if (show) {
        viewer->removeShape("ground_roi_box");
        viewer->addCube(
            -4.0, -2.3,    // X 범위
            -4.48, -2.35,  // Y 범위
            -0.2, 1.1,     // Z 범위
            1.0, 0.0, 1.0, // (R=1.0, G=1.0, B=0.0) 노란색
            "ground_roi_box"
        );
        viewer->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
            "ground_roi_box"
        );
    }
    else {
        viewer->removeShape("ground_roi_box");
    }
}

// ----------------------------------------------------------------------------
// 키보드 콜백 함수
// ----------------------------------------------------------------------------
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* viewer_void) {
    auto viewer = static_cast<pcl::visualization::PCLVisualizer*>(viewer_void);

    if (event.keyDown()) {
        if (event.getKeySym() == "p" || event.getKeySym() == "P") { // 'p' 키: 일시정지 토글
            std::lock_guard<std::mutex> lock(control_mutex);
            is_paused = !is_paused; // 일시정지 상태 토글
            if (is_paused)
                std::cout << "[INFO] Data reading paused.\n";
            else
                std::cout << "[INFO] Data reading resumed.\n";
        }
        else if (event.getKeySym() == "v" || event.getKeySym() == "V") { // 'v' 키: V_start_process 토글
            std::lock_guard<std::mutex> lock(control_mutex);
            V_start_process = !V_start_process; // V_start_process 상태 토글
            if (V_start_process)
                std::cout << "[INFO] V_start_process set to true.\n";
            else
                std::cout << "[INFO] V_start_process set to false.\n";

            // 포인트 클라우드 데이터 초기화
            resetPointCloudData();

            viewer->removeShape("v_start_process_text");
            std::string initial_status = "V_start_process: " + std::string(V_start_process ? "True" : "False");
            viewer->addText(initial_status, 20, 650, 20, 1, 1, 1, "v_start_process_text");
        }
        else if (event.getKeySym() == "c" || event.getKeySym() == "C") {
            heightCalibration_mode = !heightCalibration_mode;

            if (heightCalibration_mode) {
                try {
                    showGroundROIBox(viewer, true);
                }
                catch (const std::runtime_error& e) {
                    std::cerr << "[ERROR] " << e.what() << std::endl;
                    viewer->removeShape("ground_result_text");
                    viewer->addText("Ground Calibration Failed (No points in ROI)",
                        20, 460, 20, 1.0, 0.0, 0.0,
                        "ground_result_text");
                }
            }
            else {
                // --- OFF ---
                std::cout << "[Calibration] Ground height calibration OFF\n";
                //ground_height_fixed = false;

                viewer->removeShape("ground_result_text");
                showGroundROIBox(viewer, false);
                viewer->removePointCloud("roi_cloud");
            }
        }
    }
}

// ----------------------------------------------------------------------------
// centroid -> 어떤 스테이지인지
// ----------------------------------------------------------------------------
std::string findStageLabel(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered, std::vector<int>& stage_counts)
{
    // 초기화
    std::fill(stage_counts.begin(), stage_counts.end(), 0);

    // 각 포인트를 스테이지에 할당하여 카운트
    for (const auto& pt : cloud_filtered->points) {
        for (size_t i = 0; i < g_stages.size(); ++i) {
            float smin = std::min(g_stages[i].roi_x_start, g_stages[i].roi_x_end) * 0.001f; // mm->m
            float smax = std::max(g_stages[i].roi_x_start, g_stages[i].roi_x_end) * 0.001f; // mm->m
            if (pt.x >= smin && pt.x <= smax) {
                stage_counts[i]++;
                break; // 한 포인트는 하나의 스테이지에만 속함
            }
        }
    }

    // 가장 많은 포인트를 가진 스테이지 찾기
    int max_count = 0;
    int max_index = -1;
    for (size_t i = 0; i < stage_counts.size(); ++i) {
        if (stage_counts[i] > max_count) {
            max_count = stage_counts[i];
            max_index = i;
        }
    }

    if (max_index != -1) {
        return g_stages[max_index].label;
    }
    return "Unknown";
}

void calculatBoundingBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster, float& height, float& width, float& depth) {
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(*cluster, min_pt, max_pt);

    height = max_pt.x - min_pt.x; // 높이
    width = max_pt.z - min_pt.z; // 폭
    depth = max_pt.y - min_pt.y; // 깊이
}

// ----------------------------------------------------------------------------
// [파레트] 클러스터링
// ----------------------------------------------------------------------------
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> performClustering(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered) {

    std::vector<pcl::PointIndices> P_cluster_indices;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud_filtered);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.1);
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(200);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_filtered);
    ec.extract(P_cluster_indices);

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> P_clusters;
    P_clusters.reserve(P_cluster_indices.size());

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(P_cluster_indices.size()); ++i) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr P_cluster(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto& idx : P_cluster_indices[i].indices) {
            P_cluster->points.push_back(cloud_filtered->points[idx]);
        }
        P_cluster->width = P_cluster->points.size();
        P_cluster->height = 1;
        #pragma omp critical
        {
            P_clusters.push_back(P_cluster);
        }
    }
    return P_clusters;
}

// ----------------------------------------------------------------------------
// [파레트] 클러스터링 시각화
// ----------------------------------------------------------------------------
void visualizeClusters(
    pcl::visualization::PCLVisualizer::Ptr viewer,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& P_clusters) {

    for (size_t i = 0; i < P_clusters.size(); ++i) {
        std::string cloud_name = "cluster_" + std::to_string(i);
        std::string box_name = "cluster_box_" + std::to_string(i);
        viewer->removePointCloud(cloud_name);
        viewer->removeShape(box_name);
    }

    for (size_t i = 0; i < P_clusters.size(); ++i) {
        std::string cloud_name = "cluster_" + std::to_string(i);

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(
            P_clusters[i],
            0, 255, 0);
        viewer->addPointCloud<pcl::PointXYZ>(P_clusters[i], color_handler, cloud_name);
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloud_name);

        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*P_clusters[i], min_pt, max_pt);

        float box_x_min = min_pt.x; // x축 최소값
        float box_x_max = box_x_min + 0.15f; // 파레트 두께 15cm

        float box_y_min = min_pt.y; // 가장 멀리 있음
        float box_y_max = box_y_min + 1.0f; // 가장 가까움

        float box_z_min = min_pt.z; // 가장 왼쪽
        float box_z_max = box_z_min + 1.2f; // 가장 오른쪽

        std::string box_name = "cluster_box_" + std::to_string(i);
        viewer->addCube(
            box_x_min, box_x_max,
            box_y_min, box_y_max,
            box_z_min, box_z_max,
            1.0, 0.647, 0.0,
            box_name
        );
        viewer->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
            1.0,
            box_name
        );
    }
}

// ----------------------------------------------------------------------------
// 클러스터링 및 시각화 통합
// ----------------------------------------------------------------------------
void clusterAndVisualize(pcl::visualization::PCLVisualizer::Ptr viewer, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered) {
    auto P_clusters = performClustering(cloud_filtered);

    std::cout << "[INFO] Found Pallet " << P_clusters.size() << "\n";

    visualizeClusters(viewer, P_clusters);
}

// ----------------------------------------------------------------------------
// 클러스터 트래킹 데이터 구조 정의
// ----------------------------------------------------------------------------
struct TrackedCluster { // 클러스터 상태를 저장하는 구조체
    int id; // 고유 ID
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud; // 클러스터 포인트 클라우드
    Eigen::Vector4f centroid; // 클러스터 중심
    int missed_frames; // 누락된 프레임 수

    // Kalman Filter 상태 (옵션)
    Eigen::Vector4f state; // [x, y, z, vx, vy, vz]
    Eigen::Matrix4f covariance;
};

// ----------------------------------------------------------------------------
// [트래킹] 매니저 클래스 정의
// ----------------------------------------------------------------------------
class ClusterTracker {
public:
    ClusterTracker(float max_distance, int max_missed)
        : max_distance_(max_distance), max_missed_(max_missed), next_id_(0) {
    }

    // 현재 프레임의 클러스터들을 트래킹
    void update(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& current_clusters) {
        std::unordered_map<int, bool> matched_clusters;

        // 현재 클러스터와 기존 트랙된 클러스터 간의 거리 계산 및 매칭
        for (const auto& cluster : current_clusters) {
            Eigen::Vector4f current_centroid;
            pcl::compute3DCentroid(*cluster, current_centroid);

            float min_distance = std::numeric_limits<float>::max();
            int matched_id = -1;

            for (auto& [id, tracked] : tracked_clusters_) {
                float distance = (current_centroid - tracked.centroid).norm();
                if (distance < min_distance && distance < max_distance_) {
                    min_distance = distance;
                    matched_id = id;
                }
            }

            if (matched_id != -1) {
                // 매칭된 경우
                tracked_clusters_[matched_id].cloud = cluster;
                tracked_clusters_[matched_id].centroid = current_centroid;
                tracked_clusters_[matched_id].missed_frames = 0;
                matched_clusters[matched_id] = true;
            }
            else {
                // 새로운 클러스터
                TrackedCluster new_cluster;
                new_cluster.id = next_id_++;
                new_cluster.cloud = cluster;
                pcl::compute3DCentroid(*cluster, new_cluster.centroid);
                new_cluster.missed_frames = 0;
                tracked_clusters_[new_cluster.id] = new_cluster;
                matched_clusters[new_cluster.id] = true;
            }
        }

        // 누락된 트랙 업데이트
        for (auto& [id, tracked] : tracked_clusters_) {
            if (matched_clusters.find(id) == matched_clusters.end()) {
                tracked.missed_frames++;
            }
        }

        // 누락된 프레임이 일정 수 이상인 트랙 삭제
        std::vector<int> to_remove;
        for (const auto& [id, tracked] : tracked_clusters_) {
            if (tracked.missed_frames > max_missed_) {
                to_remove.push_back(id);
            }
        }
        for (const auto& id : to_remove) {
            tracked_clusters_.erase(id);
        }
    }

    // 트랙된 클러스터들 반환
    const std::unordered_map<int, TrackedCluster>& getTrackedClusters() const {
        return tracked_clusters_;
    }

private:
    float max_distance_; // 매칭 최대 거리
    int max_missed_; // 최대 누락 프레임 수
    int next_id_; // 다음 클러스터 ID

    std::unordered_map<int, TrackedCluster> tracked_clusters_;
};

// ----------------------------------------------------------------------------
// [트래킹] 파레트 클러스터링 시각화
// ----------------------------------------------------------------------------
void visualizeTrackedClusters(
    pcl::visualization::PCLVisualizer::Ptr viewer,
    const std::unordered_map<int, TrackedCluster>& tracked_clusters) {

    // 기존 클러스터 시각화 제거
    for (const auto& [id, tracked] : tracked_clusters) {
        std::string cloud_name = "cluster_" + std::to_string(id);
        std::string box_name = "cluster_box_" + std::to_string(id);
        viewer->removePointCloud(cloud_name);
        viewer->removeShape(box_name);
    }

    cluster_box_ids.clear();

    for (const auto& [id, tracked] : tracked_clusters) {
        std::string cloud_name = "cluster_" + std::to_string(id);
        std::string box_name = "cluster_box_" + std::to_string(id);

        // 클러스터 포인트 클라우드 추가 (색상 고정 또는 ID 기반 색상)
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(
            tracked.cloud,
            0,   // 빨강 값 (녹색으로 표시)
            255, // 녹색 값
            0    // 파랑 값
        );
        viewer->addPointCloud<pcl::PointXYZ>(tracked.cloud, color_handler, cloud_name);
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloud_name);

        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*tracked.cloud, min_pt, max_pt);

        // 클러스터의 x축 중심 계산
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*tracked.cloud, centroid);
        float centroid_x = centroid[0];

        // 박스의 x축 범위 계산 (centroid_x를 중심으로 두께 0.15m)
        float box_x_min = centroid_x - 0.075f; // 0.15m의 절반
        float box_x_max = centroid_x + 0.075f;

        float box_y_min = min_pt.y; // 가장 멀리 있음
        float box_y_max = box_y_min + 1.0f; // 가장 가까움

        float box_z_min = min_pt.z; // 가장 왼쪽
        float box_z_max = box_z_min + 1.2f; // 가장 오른쪽

        viewer->addCube(
            box_x_min, box_x_max,
            box_y_min, box_y_max,
            box_z_min, box_z_max,
            1.0, 0.647, 0.0,
            box_name
        );
        viewer->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
            box_name
        );
        viewer->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
            1.0,
            box_name
        );

        cluster_box_ids.push_back(box_name);

    }
}

// ----------------------------------------------------------------------------
// [부피 측정] 평균 계산
// ----------------------------------------------------------------------------
float calculateAverage(std::vector<float>& values) {
    std::sort(values.begin(), values.end());

    int num_remove = static_cast<int>(values.size() * 0.2);
    float sum = 0.0f;
    int count = 0;

    for (size_t i = num_remove; i < values.size() - num_remove; ++i) {
        sum += values[i];
        count++;
    }
    return (count > 0) ? (sum / count) : 0.0f;
}
float calculateAverageX(std::vector<float>& x_lengths) {
    float average_x = 0;
    if (x_lengths.size() == vector_size) {
        average_x = calculateAverage(x_lengths);
    }
    return average_x;
}
float calculateAverageY(std::vector<float>& y_lengths) {
    float average_y = 0;
    if (y_lengths.size() == vector_size) {
        average_y = calculateAverage(y_lengths);
    }
    return average_y;
}
float calculateAverageZ(std::vector<float>& z_lengths) {
    float average_z = 0;
    if (z_lengths.size() == vector_size) {
        average_z = calculateAverage(z_lengths);
    }
    return average_z;
}

// ----------------------------------------------------------------------------
// [부피 측정] YZ 평면 클러스터링
// ----------------------------------------------------------------------------
void detectPlaneYZ(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::visualization::PCLVisualizer::Ptr viewer) {
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ExtractIndices<pcl::PointXYZ> extract;

    float max_length_x = 0.0;
    float max_length_y = 0.0;
    float max_length_z = 0.0;
    pcl::PointXYZ p1_x, p2_x;
    pcl::PointXYZ p1_y, p2_y;
    pcl::PointXYZ p1_z, p2_z;
    try {
        while (true) {
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.06);
            seg.setInputCloud(cloud);
            seg.segment(*inliers, *coefficients);

            if (inliers->indices.size() < 100) {
                std::cout << "No more planes found." << std::endl;
                break;
            }

            // 평면의 포인트 클라우드 생성
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
            extract.setInputCloud(cloud);
            extract.setIndices(inliers);
            extract.setNegative(false);
            extract.filter(*cloud_plane);

            // 클러스터링
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
            tree->setInputCloud(cloud_plane);

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
            ec.setClusterTolerance(0.06);
            ec.setMinClusterSize(150);
            ec.setMaxClusterSize(3000);
            ec.setSearchMethod(tree);
            ec.setInputCloud(cloud_plane);
            ec.extract(cluster_indices);

            for (const auto& indices : cluster_indices) {
                float min_x = std::numeric_limits<float>::max();
                float max_x = std::numeric_limits<float>::lowest();
                float min_y = std::numeric_limits<float>::max();
                float max_y = std::numeric_limits<float>::lowest();
                float min_z = std::numeric_limits<float>::max();
                float max_z = std::numeric_limits<float>::lowest();

                for (const auto& index : indices.indices) {
                    const auto& point = cloud_plane->points[index];
                    min_x = std::min(min_x, point.x);
                    max_x = std::max(max_x, point.x);
                    min_y = std::min(min_y, point.y);
                    max_y = std::max(max_y, point.y);
                    min_z = std::min(min_z, point.z);
                    max_z = std::max(max_z, point.z);
                }

                float length_x = max_x - min_x;
                if (length_x > max_length_x) {
                    max_length_x = length_x;
                    p1_x = pcl::PointXYZ(min_x, min_y, min_z);
                    p2_x = pcl::PointXYZ(max_x, min_y, min_z);
                }

                std::cout << "max_y : " << max_y << " min_y : " << min_y << std::endl;
                float length_y = max_y - min_y; // Y축 방향 길이
                if (length_y > max_length_y) {
                    max_length_y = length_y;
                    p1_y = pcl::PointXYZ(min_x, min_y, min_z);
                    p2_y = pcl::PointXYZ(min_x, max_y, min_z);
                }

                float length_z = max_z - min_z; // Z축 방향 길이
                if (length_z > max_length_z) {
                    max_length_z = length_z;
                    p1_z = pcl::PointXYZ(min_x, min_y, min_z);
                    p2_z = pcl::PointXYZ(min_x, min_y, max_z);
                }
            }

            extract.setNegative(true);
            extract.filter(*cloud);
        }
    }   
	catch (const std::exception& e) {
		//std::cerr << e.what() << std::endl;
	}
    if (max_length_y > 0) {
        std::string line_id_y = "longest_line_y";

        viewer->removeShape(line_id_y);
        pcl::PointXYZ start_y(p1_y.x, p1_y.y, p1_y.z);
        pcl::PointXYZ end_y(p1_y.x, p1_y.y + max_length_y, p1_y.z);
        viewer->addLine(start_y, end_y, 0.0, 1.0, 0.0, line_id_y);

        volume_line_ids.push_back(line_id_y);

        bool is_duplicate = false;
        for (size_t i = 0; i < y_lengths.size(); ++i) {
            if (y_lengths[i] == max_length_y) {
                is_duplicate = true;
                break;
            }
        }

        if (!is_duplicate) {
            std::cout << "Length" + std::to_string(y_index) + " : " + std::to_string(static_cast<int>(max_length_y * 1000)) << std::endl;
            saveToFile("Length" + std::to_string(y_index) + " : " + std::to_string(static_cast<int>(max_length_y * 1000)));

            if (y_lengths.size() < vector_size) {
                y_lengths.push_back(max_length_y);
            }
            else {
                y_lengths[y_index] = max_length_y;
            }
            y_index = (y_index + 1) % vector_size;
        }
    }

    if (max_length_z > 0) {
        std::string line_id_z = "longest_line_z";
        viewer->removeShape(line_id_z);
        pcl::PointXYZ start_z(p1_z.x, p1_z.y, p1_z.z);
        pcl::PointXYZ end_z(p1_z.x, p1_z.y, p1_z.z + max_length_z);
        viewer->addLine(start_z, end_z, 0.0, 0.0, 1.0, line_id_z);

        volume_line_ids.push_back(line_id_z);

        bool is_duplicate = false;
        for (size_t i = 0; i < z_lengths.size(); ++i) {
            if (z_lengths[i] == max_length_z) {
                is_duplicate = true;
                break;
            }
        }

        if (!is_duplicate) {
            std::cout << "Width" + std::to_string(z_index) + " : " + std::to_string(static_cast<int>(max_length_z * 1000)) << std::endl;
            saveToFile("Width" + std::to_string(z_index) + " : " + std::to_string(static_cast<int>(max_length_z * 1000)));

            if (z_lengths.size() < vector_size) {
                z_lengths.push_back(max_length_z);
            }
            else {
                z_lengths[z_index] = max_length_z;
            }
            z_index = (z_index + 1) % vector_size;
        }
    }
}

// ----------------------------------------------------------------------------
// [부피 측정] 높이 계산
// ----------------------------------------------------------------------------
void calcMaxX(std::vector<float>& x_values, float& max_x_value)
{
    int total_size = x_values.size();

    if (total_size > 100)
    {
        std::sort(x_values.begin(), x_values.end(), std::greater<float>());
        x_values.resize(100);

        int remove_count = x_values.size() * 0.2;
        std::vector<float> x_values_vec(x_values.begin() + remove_count, x_values.end() - remove_count);

        if (x_values_vec.size() >= 60)
        {
            float sum_x = 0.0f;
            for (float x : x_values_vec)
            {
                //std::cout << x << " ";
                sum_x += x;
            }

            max_x_value = sum_x / x_values_vec.size();
            std::cout << "Height" + std::to_string(x_index) + " : " + std::to_string(static_cast<int>(max_x_value * 1000)) << std::endl;

            if (x_lengths.size() < vector_size) {
                x_lengths.push_back(max_x_value);
            }
            else {
                x_lengths[x_index] = max_x_value;
            }

            x_index = (x_index + 1) % vector_size;
        }
        else
        {
            max_x_value = 0;
        }
    }
    else
    {
        max_x_value = 0;
    }
}

// ----------------------------------------------------------------------------
// [부피 측정] 기울기 계산
// ----------------------------------------------------------------------------
void calculateAnglePoints(const pcl::PointXYZ& start_point, const pcl::PointXYZ& end_point, pcl::visualization::PCLVisualizer::Ptr viewer) {
    float dy = end_point.y - start_point.y;
    float angle_radians = std::atan(dy);  // 기울기


    angle_degrees = angle_radians * 180.0 / M_PI;

    std::cout << "start_min_y_point: ("
        << start_point.x << ", " << start_point.y << ", " << start_point.z << ")" << std::endl;
    std::cout << "end_min_y_point: ("
        << end_point.x << ", " << end_point.y << ", " << end_point.z << ")" << std::endl;


    viewer->removeShape("angle_line");
    pcl::PointXYZ start_z(start_point.x, start_point.y, start_point.z);
    pcl::PointXYZ end_z(start_point.x, end_point.y, end_point.z);
    viewer->addLine(start_z, end_z, 0.0, 1.0, 1.0, "angle_line");
}

// ----------------------------------------------------------------------------
// [높이 센서] 적재물 식별 및 높이 계산 함수
// ----------------------------------------------------------------------------
PalletInfo identifyPallet(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    PalletInfo info;
    info.is_pallet = false;
    info.P_height = 0.0f;

    int count_first_roi = 0;
    float min_x_first_roi = std::numeric_limits<float>::max();

    for (const auto& point : cloud->points) {
        if (point.y >= -1600.0f && point.y <= -800.0f &&
            point.z >= 0.0f && point.z <= 1000.0f) {
            count_first_roi++;
            if (point.x < min_x_first_roi) {
                min_x_first_roi = point.x;
            }
        }
    }
    if (count_first_roi >= 20 && count_first_roi <= 400) {
        info.is_pallet = true;
        info.P_height = min_x_first_roi; // mm 단위
        return info;
    }
    int count_second_roi = 0;
    float min_x_second_roi = std::numeric_limits<float>::max();

    for (const auto& point : cloud->points) {
        if (point.y >= -300.0f && point.y <= 0.0f &&
            point.z >= -100.0f && point.z <= 200.0f) {
            count_second_roi++;
            if (point.x < min_x_second_roi) {
                min_x_second_roi = point.x;
            }
        }
    }
    if (count_second_roi >= 20 && count_second_roi <= 80) {
        info.is_pallet = true;
        info.P_height = min_x_second_roi - 700.0f;
        return info;
    }
    return info;
}

// ----------------------------------------------------------------------------
// [높이 센서] 시각화
// ----------------------------------------------------------------------------

void visualizeHeight(pcl::visualization::PCLVisualizer::Ptr viewer, const PalletInfo& pallet_info, int index) {
    if (!pallet_info.is_pallet) return;

    std::string line_id = "pallet_line_" + std::to_string(index);
    viewer->removeShape(line_id);

    pcl::PointXYZ start_pallet(0.0f, 0.0f, 0.0f);
    pcl::PointXYZ end_pallet(pallet_info.P_height, 0.0f, 0.0f);

    viewer->addLine(start_pallet, end_pallet, 0.0, 0.0, 1.0, line_id);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, line_id);
    pallet_line_ids.push_back(line_id);

    std::string text_id = "pallet_height_text_" + std::to_string(index);
    viewer->removeShape(text_id);

    std::stringstream ss_height;
    ss_height << "Height: " << pallet_info.P_height << " mm";

    viewer->addText(ss_height.str(), 20, 220 + index * 20, 20, 1, 1, 1, text_id);
    pallet_height_text_ids.push_back(text_id);
}

// ----------------------------------------------------------------------------
// 이전 파레트 선 및 텍스트 제거 함수
// ----------------------------------------------------------------------------
void removePreviousPalletVisualizations(pcl::visualization::PCLVisualizer::Ptr viewer) {
    // 기존의 선 제거
    for (const auto& line_id : pallet_line_ids) {
        viewer->removeShape(line_id);
    }
    pallet_line_ids.clear();

    // 기존의 텍스트 제거
    for (const auto& text_id : pallet_height_text_ids) {
        viewer->removeShape(text_id);
    }
    pallet_height_text_ids.clear();
}

// ----------------------------------------------------------------------------
// 클러스터링 및 파레트 식별 통합 함수 (포인트 기반)
// ----------------------------------------------------------------------------
void processPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_merge,
    pcl::visualization::PCLVisualizer::Ptr viewer,
    int& previous_index) {
    PalletInfo pallet_info = identifyPallet(cloud_merge);

    if (pallet_info.is_pallet) {
        std::cout << "Pallet Height: " << pallet_info.P_height << " mm" << std::endl;
        // 필요한 경우 JSON 생성 및 ZMQ 전송 로직 추가
        // ...

        // 시각화: 파레트 높이 선 및 텍스트 추가
        visualizeHeight(viewer, pallet_info, previous_index);
        previous_index++;
    }
}




// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, const char* argv[]) {

    // [트래킹] 초기화
    ClusterTracker tracker(2.5f, 300); // 최대 매칭 거리 1.0m, 최대 누락 프레임 수 5

    std::cout << "Current working directory: "
        << std::filesystem::current_path() << std::endl;

    // 1) 설정 로드
    const std::string LIVOX_PATH = "config/config.json";
    const std::string WATA_PATH = "config/setting.json";
    WATAConfig config = readConfigFromJson(WATA_PATH);

    bool READ_PCD_FROM_FILE = config.read_file;
    bool SAVE_PCD_FROM_FILE = config.save_file;
    const std::string READ_PCD_FILE_NAME = config.read_file_name;
    const std::string SAVE_PCD_FILE_NAME = config.save_file_name;

    std::cout << "[DEBUG] Found " << g_stages.size() << " stages in JSON.\n";
    for (auto& st : g_stages) {
        std::cout << "  " << st.label << ": x_start=" << st.roi_x_start
            << ", x_end=" << st.roi_x_end << std::endl;
    }    

    const int iteration = config.iteration;

    float V_x_min = std::min(config.V_x_start, config.V_x_end) * 0.001f;
    float V_x_max = std::max(config.V_x_start, config.V_x_end) * 0.001f;

    std::cout << "[debug] roi_y_start: " << config.roi_y_start << std::endl;
    std::cout << "[debug] roi_z_start: " << config.roi_z_start << std::endl;

    // 3) y,z ROI => mm->m
    float y_min = std::min(config.roi_y_start, config.roi_y_end) * 0.001f; // ex) -1.6
    float y_max = std::max(config.roi_y_start, config.roi_y_end) * 0.001f; // ex) -0.3
    float z_min = std::min(config.roi_z_start, config.roi_z_end) * 0.001f;
    float z_max = std::max(config.roi_z_start, config.roi_z_end) * 0.001f;

    std::cout << "[debug] ROI y_min: " << y_min << ", y_max: " << y_max << std::endl;
    std::cout << "[debug] ROI z_min: " << z_min << ", z_max: " << z_max << std::endl;

    const float ANGLE_DEGREES = config.angle;
    const float THETA = ANGLE_DEGREES * M_PI / 180.f;
    const float COS_THETA = std::cos(THETA);
    const float SIN_THETA = std::sin(THETA);

    if (READ_PCD_FROM_FILE) {
        // 파일에서 PCD 로드
        std::cout << "Reading point cloud: " << READ_PCD_FILE_NAME << std::endl;
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(READ_PCD_FILE_NAME, *cloud_loaded) == -1) {
            PCL_ERROR("Could not read file\n");
            return -1;
        }
        std::cout << "[INFO] Loaded " << cloud_loaded->size()
            << " points from " << READ_PCD_FILE_NAME << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    else {
        // 실시간 (Livox) 모드
        if (!LivoxLidarSdkInit(LIVOX_PATH.c_str())) {
            std::cerr << "[ERROR] Livox Init Failed\n";
            LivoxLidarSdkUninit();
            return -1;
        }
        SetLivoxLidarPointCloudCallBack(PointCloudCallback, &config);
    }

    // ZMQ
    zmq::context_t context(1);
    zmq::socket_t publisher(context, ZMQ_PUB);
    publisher.bind("tcp://127.0.0.1:5001");

    zmq::socket_t subscriber(context, ZMQ_SUB);
    subscriber.connect("tcp://127.0.0.1:5002");
    subscriber.set(zmq::sockopt::subscribe, "LIS>MID360");

    // 4) Viewer
    auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>("3D Viewer");
    viewer->addCoordinateSystem(1.0);
    viewer->setBackgroundColor(0.1, 0.1, 0.1);
    viewer->setCameraPosition(8.88232, 11.3493, -8.39895, 0.946026, -0.261667, 0.191218);

    int previous_pallet_index = 0;

    viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)viewer.get());


    x_lengths.clear();
    y_lengths.clear();
    z_lengths.clear();

    // V_start_process 상태 텍스트 초기화
    std::string initial_status = "V_start_process: " + std::string(V_start_process ? "True" : "False");
    viewer->addText(initial_status, 20, 650, 20, 1, 1, 1, "v_start_process_text");

    // 4-1) 스테이지별 박스 시각화
    for (size_t i = 0; i < g_stages.size(); ++i) {
        // stageX in [ x_min, x_max ]
        double x1 = (double)std::min(g_stages[i].roi_x_start, g_stages[i].roi_x_end) * 0.001; // mm->m
        double x2 = (double)std::max(g_stages[i].roi_x_start, g_stages[i].roi_x_end) * 0.001;
        double y1 = (double)y_min;
        double y2 = (double)y_max;
        double z1 = (double)z_min;
        double z2 = (double)z_max;

        double R = 1.0, G = 1.0, B = 0.0; // 노랑
        std::string shape_id = "roi_box_" + g_stages[i].label;
        viewer->addCube(
            x1, x2,
            y1, y2,
            z1, z2,
            R, G, B,
            shape_id
        );
        viewer->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
            shape_id
        );
        viewer->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
            1.0,
            shape_id
        );
    }

    // 전역 변수 또는 메인 루프 상단에 추가
    bool previous_V_start_process = V_start_process;

    // 메인 루프
    while (!viewer->wasStopped()) {
        try {
            viewer->spinOnce();
            updateCameraPositionText(viewer);

            {
                std::lock_guard<std::mutex> lock(control_mutex);
                if (is_paused) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 짧게 대기
                    continue; // 루프의 나머지 부분을 건너뜀
                }
            }

            // 2) 파일 모드: chunk 로딩
            if (READ_PCD_FROM_FILE) {
                std::lock_guard<std::mutex> lk(control_mutex);
                if (reading_active) {
                    // chunk_size = 96 * iteration = 96 * 100 = 9,600
                    const int chunk_size = 96 * iteration;
                    int count_pushed = 0;

                    while (point_index < static_cast<int>(cloud_loaded->points.size()) && count_pushed < chunk_size) {
                        cloud_merge->points.push_back(cloud_loaded->points[point_index]);
                        ++point_index;
                        ++count_pushed;
                    }
                    cloud_merge->width = cloud_merge->points.size();
                    cloud_merge->height = 1;

                    //if (count_pushed > 0) {
                    //    V_start_process = true;
                    //}
                }
            }

            if (heightCalibration_mode) {
                // 1.1) ROI 내의 지면 포인트만 추출
                {
                    std::lock_guard<std::mutex> lock(g_mutex);

                    cloud_ground->clear();

                    float x_min = -4.0f, x_max = -2.3f;
                    float y_min = -4.48f, y_max = -2.35f;
                    float z_min = -0.2f, z_max = 1.1f;

                    pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud;
                    if (READ_PCD_FROM_FILE) {
                        src_cloud = cloud_loaded;
                    }
                    else {
                        src_cloud = cloud_raw;
                    }

                    for (const auto& pt : src_cloud->points) {
                        if (pt.x >= x_min && pt.x <= x_max &&
                            pt.y >= y_min && pt.y <= y_max &&
                            pt.z >= z_min && pt.z <= z_max)
                        {
                            cloud_ground->push_back(pt);
                        }
                    }
                    cloud_ground->width = static_cast<uint32_t>(cloud_ground->points.size());
                    cloud_ground->height = 1;

                    // 1.2) Voxel
                    voxelizePointCloud(cloud_ground, 0.1f, 0.1f, 0.1f);
                }

                // 1.3) 지면높이 계산
                try {
                    fixed_ground_height = std::abs(calculateGroundHeight(cloud_ground));
                    ground_height_fixed = true;

                    std::cout << "[Calibration] Ground height fixed: " << fixed_ground_height << " m" << std::endl;

                    viewer->removeShape("ground_result_text");
                    std::ostringstream oss;
                    oss << "Fixed Ground Height: " << fixed_ground_height << " m";
                    viewer->addText(
                        oss.str(),
                        20, 460, 20,
                        0.0, 1.0, 0.0,
                        "ground_result_text"
                    );


                    // 1.4) 시각화용으로 ROI 포인트 보여주기
                    viewer->removePointCloud("roi_cloud");
                    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
                        color_handler(cloud_ground, 255, 0, 0);
                    viewer->addPointCloud<pcl::PointXYZ>(cloud_ground, color_handler, "roi_cloud");
                    viewer->setPointCloudRenderingProperties(
                        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "roi_cloud"
                    );
                }
                catch (const std::runtime_error& e) {
                    std::cerr << "[ERROR] " << e.what() << std::endl;
                    viewer->removeShape("ground_result_text");
                    viewer->addText("Ground Calibration Failed (No points in ROI)",
                        20, 460, 20, 1.0, 0.0, 0.0,
                        "ground_result_text");
                }

                // 1.5) 한 번 계산이 끝났으면, 다음번엔 다시 계산하지 않도록 한다면
                // heightCalibration_mode = false;  
                // (혹은 키보드 콜백에서 꺼지도록 하거나, 원하는 로직에 맞춰 처리가능)
            }


            // 3) ZMQ 메시지 수신 (측정 시작/중지)
            zmq::message_t msg;
            if (subscriber.recv(msg, zmq::recv_flags::dontwait)) {
                std::string message(static_cast<char*>(msg.data()), msg.size());
                if (message == "LIS>MID360,1") {
                    V_start_process = true;
                }
                else if (message == "LIS>MID360,0") {
                    V_start_process = false;
                }
                else if (message == "LIS>MID360,2") {
                    heightCalibration_mode = true;
                }
                saveToFile("[" + message + "], timestamp: " + getCurrentTime());
            }

            // ------------------------------------------------------------------
            // 상태 전환 감지 및 처리
            // ------------------------------------------------------------------
            if (V_start_process != previous_V_start_process) {
                // 이전 상태에 따른 Viewer 데이터 제거
                if (previous_V_start_process) {
                    // 이전 상태가 true였을 때: 부피 형상 측정 관련 데이터 제거
                    viewer->removePointCloud("cloud_pcd");
                    viewer->removePointCloud("cloud_angle_filtered");
                    viewer->removeShape("result");
                    viewer->removeShape("angle_line");
                    for (const auto& line_id : volume_line_ids) {
                        viewer->removeShape(line_id);
                    }
                    volume_line_ids.clear();
                }
                else {
                    // 이전 상태가 false였을 때: 클러스터링 및 스테이지 카운팅 관련 데이터 제거
                    viewer->removePointCloud("filtered_cloud");
                    viewer->removeShape("stageText");
                    // 클러스터별로 추가된 포인트 클라우드와 박스 제거
                    for (size_t i = 0; i < g_stages.size(); ++i) {
                        std::string count_text_id = "stage" + std::to_string(i + 1) + "_count";
                        viewer->removeShape(count_text_id);
                    }
                    // 파레트 클러스터링 3D 박스 제거
                    for (const auto& box_id : cluster_box_ids) {
                        viewer->removeShape(box_id);
                                        }
                    // 트래킹 클러스터 제거
                    viewer->removeAllPointClouds(); // 모든 클러스터 포인트 클라우드 제거 (옵션)
                    
                }

                // 포인트 클라우드 데이터 초기화
                resetPointCloudData();

                // 이전 상태 업데이트
                previous_V_start_process = V_start_process;

                // Viewer에 현재 상태 텍스트 업데이트
                std::string status_text = "V_start_process: " + std::string(V_start_process ? "True" : "False");
                viewer->removeShape("v_start_process_text");
                viewer->addText(status_text, 20, 650, 20, 1, 1, 1, "v_start_process_text");

                std::cout << "[INFO] V_start_process state changed to "
                    << (V_start_process ? "True" : "False") << ". Data has been reset." << std::endl;
            }

            if (cloud_merge && !cloud_merge->empty()) {
                std::lock_guard<std::mutex> lk(g_mutex);
                // -------------------------------------------------------------------------------------
                // 부피 형상 측정
                // -------------------------------------------------------------------------------------
                if (V_start_process) {
                    // V_start_process가 true인 경우: x < 0.0f인 포인트 처리
                    float start_min_y = std::numeric_limits<float>::infinity();
                    pcl::PointXYZ start_min_y_point;
                    float end_min_y = std::numeric_limits<float>::infinity();
                    pcl::PointXYZ end_min_y_point;
                    std::vector<float> x_values;

                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcd_local(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_volume(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_angle_filtered(new pcl::PointCloud<pcl::PointXYZ>);

                    for (auto& temp : cloud_merge->points) {
                        pcl::PointXYZ point;
                        point.x = temp.x;
                        point.y = temp.y * COS_THETA - temp.z * SIN_THETA;
                        point.z = temp.y * SIN_THETA + temp.z * COS_THETA;

                        if (point.x < V_x_max && point.x >= V_x_min &&
                            point.y >= y_min && point.y <= y_max &&
                            point.z >= z_min && point.z <= z_max) {
                            x_values.push_back(point.x);
                            cloud_filtered_volume->points.push_back(point);
                            cloud_pcd_local->points.push_back(point);

                            if (point.z >= 0.1f && point.z <= 0.3f) {
                                if (point.y < start_min_y) {
                                    start_min_y = point.y;
                                    start_min_y_point = point;
                                }
                            }

                            if (point.z > 0.7f && point.z <= 0.9f) {
                                if (point.y < end_min_y) {
                                    end_min_y = point.y;
                                    end_min_y_point = point;
                                }
                            }
                        }
                    }

                    cloud_filtered_volume->width = cloud_filtered_volume->size();
                    cloud_filtered_volume->height = 1;

                    if (!x_values.empty()) {
                        calcMaxX(x_values, max_x_value);
                        for (auto& temp : cloud_filtered_volume->points) {
                            temp.x = max_x_value;
                        }
                    }

               
                    // Voxel Downsample
                    voxelizePointCloud(cloud_filtered_volume, 0.03f, 0.03f, 0.03f);
                    // Outlier Remove
                    removeOutliers(cloud_filtered_volume, config);

                    // Angle Points 계산
                    calculateAnglePoints(start_min_y_point, end_min_y_point, viewer);

                    // 추가 각도 보정 (재계산 필요)
                    float COS_THETA_updated = cos(angle_degrees * M_PI / 180.0);
                    float SIN_THETA_updated = sin(angle_degrees * M_PI / 180.0);
                    for (auto& temp : cloud_filtered_volume->points) {
                        pcl::PointXYZ point;
                        point.x = temp.x;
                        point.y = temp.y * COS_THETA_updated - temp.z * SIN_THETA_updated;
                        point.z = temp.y * SIN_THETA_updated + temp.z * COS_THETA_updated;
                        cloud_angle_filtered->points.push_back(point);
                    }

                    viewer->removePointCloud("cloud_angle_filtered");
                    viewer->addPointCloud<pcl::PointXYZ>(cloud_angle_filtered, "cloud_angle_filtered");

                    detectPlaneYZ(cloud_angle_filtered, viewer);

                    // 부피 측정 결과 계산
                    if (x_lengths.size() == vector_size) {
                        result_height = calculateAverageX(x_lengths) * 1000;
                        x_lengths.clear();
                    }

                    if (y_lengths.size() == vector_size) {
                        result_length = calculateAverageY(y_lengths) * 1000;
                        y_lengths.clear();
                    }

                    if (z_lengths.size() == vector_size) {
                        result_width = calculateAverageZ(z_lengths) * 1000;
                        z_lengths.clear();
                    }

                    if (result_height != 0 && result_width != 0 && result_length != 0) {
                        bool result_status = true;
                        
                        float ground_correction_mm = 0.0f;
                        if (ground_height_fixed) {
                            ground_correction_mm = fixed_ground_height * 1000.0f;
                            result_height += (ground_correction_mm - 117.0f);
                        }
                        else {
                            result_height += config.height_threshold;  // 기존 임시 보정 (2755)
                        }

                        std::string json_result = "{"
                            "\"height\": " + std::to_string(result_height) + ", "
                            "\"width\": " + std::to_string(result_width) + ", "
                            "\"length\": " + std::to_string(result_length) + ", "
                            "\"result\": " + std::to_string(result_status) + ", "
                            "\"timestamp\": \"" + getCurrentTime() + "\", "
                            "\"points\": [";

                        voxelizePointCloud(cloud_pcd_local, 0.03, 0.03, 0.03);

                        viewer->removePointCloud("cloud_pcd");
                        viewer->addPointCloud<pcl::PointXYZ>(cloud_pcd_local, "cloud_pcd");

                        for (size_t i = 0; i < cloud_pcd_local->points.size(); ++i) {
                            json_result += "{"
                                "\"x\": " + std::to_string(cloud_pcd_local->points[i].x) + ", "
                                "\"y\": " + std::to_string(cloud_pcd_local->points[i].y) + ", "
                                "\"z\": " + std::to_string(cloud_pcd_local->points[i].z) +
                                "}";
                            if (i < cloud_pcd_local->points.size() - 1) {
                                json_result += ", ";
                            }
                        }

                        json_result += "] }";

                        std::ostringstream oss;
                        oss << "Height: " << result_height << " mm \n"
                            << "Width: " << result_width << " mm \n"
                            << "Length: " << result_length << " mm \n"
                            << "Angle: " << angle_degrees << " deg \n"
                            << "GroundFix: " << ground_correction_mm << " mm \n" // 추가
                            << "PCD: " << cloud_pcd_local->points.size() << " cnt ";

                        std::string result = oss.str();

                        viewer->removeShape("result");
                        viewer->addText(result.c_str(), 20, 320, 20, 1, 1, 1, "result");

                        std::string msg_pub = "MID360>LIS " + json_result;
                        zmq::message_t topic_msg(msg_pub.c_str(), msg_pub.length());
                        publisher.send(topic_msg, zmq::send_flags::dontwait);

                        std::cout << "[LOG] " << result << std::endl;

                        saveToFile("[SEND]" + result);

                        result_height = 0;
                        result_width = 0;
                        result_length = 0;
                    }

                }
                else {
                    // V_start_process가 false인 경우: x >= 0.0f인 포인트 처리
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

                    for (auto& pt : cloud_merge->points) {
                        float px = pt.x;
                        float py = pt.y;
                        float pz = pt.z;

                        float x_min_all = -2.995f;
                        float x_max_all = 5.61f;

                        if (px >= x_min_all &&
                            px <= x_max_all &&
                            py >= y_min && py <= y_max &&
                            pz >= z_min && pz <= z_max)
                        {
                            cloud_filtered->push_back(pt);
                        }
                    }
                    cloud_filtered->width = cloud_filtered->size();
                    cloud_filtered->height = 1;



                    // Voxel Downsample
                    voxelizePointCloud(cloud_filtered, 0.05f, 0.05f, 0.05f);

                    // Outlier Remove
                    removeOutliers(cloud_filtered, config);

                    //static int previous_index = 0;
                    //PalletInfo pallet_info = identifyPallet(cloud_filtered);
                    //if (pallet_info.is_pallet) {
                    //    visualizeHeight(viewer, pallet_info, previous_index);
                    //    previous_index++;
                    //}

                    //processPoints(cloud_merge, viewer, previous_pallet_index);


                    //// 클러스터링 및 트래킹 (V_start_process == false)
                    //std::future<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> future_clusters =
                    //    std::async(std::launch::async, performClustering, cloud_filtered);

                    //// 클러스터링 수행
                    //auto P_clusters = future_clusters.get();
                    //// 트래커 업데이트
                    //tracker.update(P_clusters);
                    //// 시각화
                    //visualizeTrackedClusters(viewer, tracker.getTrackedClusters());

                    // Stage 판별 (포인트 카운팅 방식 사용)
                    std::string current_stage = "Unknown";
                    std::vector<int> stage_counts(g_stages.size(), 0);
                    if (!cloud_filtered->empty()) {
                        current_stage = findStageLabel(cloud_filtered, stage_counts);
                    }

                    viewer->removePointCloud("filtered_cloud");
                    viewer->addPointCloud<pcl::PointXYZ>(cloud_filtered, "filtered_cloud");

                    viewer->removeShape("stageText");
                    viewer->addText("Current Stage: " + current_stage,
                        20, 40, 20, 1, 1, 1, "stageText");

                    // 텍스트 표시: Stage별 포인트 개수
                    for (size_t i = 0; i < g_stages.size(); ++i) {
                        std::string count_text_id = "stage" + std::to_string(i + 1) + "_count";
                        viewer->removeShape(count_text_id);
                        viewer->addText("Stage " + std::to_string(i + 1) + " Points: " + std::to_string(stage_counts[i]),
                            20, 60 + static_cast<int>(i) * 20, // y 위치: 60, 80, 100, 120
                            20, // 글자 크기
                            1, 1, 1, // 흰색
                            count_text_id);
                    }

                    // PCD 데이터 저장 (단순 누적)
                    if (SAVE_PCD_FROM_FILE) {
                        *cloud_pcd += *cloud_merge;
                        std::cout << "[DEBUG] Accumulated " << cloud_pcd->size() << " points in cloud_pcd. \n";
                    }

                    // 파일 모드라면 다음 chunk 대기 위해 처리 종료
                    if (READ_PCD_FROM_FILE) {
                        cloud_merge->clear();
                    }
                }

                // 클라우드 처리 후 클라우드_merge 초기화
                cloud_merge->clear();
            }
        }
        catch (const std::exception& e) {
            std::string error_message = "[ERROR] Exception: ";
            error_message += e.what();
            std::cerr << error_message << std::endl;
            saveToFile(error_message);
        }
    }
    // 종료 시 PCD 저장
    if (SAVE_PCD_FROM_FILE && !cloud_pcd->empty()) {
        std::cout << "[INFO] Saving point cloud... " << std::endl;
        if (pcl::io::savePCDFileBinary(SAVE_PCD_FILE_NAME, *cloud_pcd) == 1) {
            PCL_ERROR("Failed to save PCD file to %s. \n", SAVE_PCD_FILE_NAME.c_str());
        }
        else {
            std::cout << "[INFO] Saved " << cloud_pcd->size()
                << " points to " << SAVE_PCD_FILE_NAME << std::endl;
        }
    }

    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>[STOP DETECTION]>>>>>>>>>>>>>>>>>>>>>>>>\n";
    LivoxLidarSdkUninit();
    return 0;
}
