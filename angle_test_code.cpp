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
    int roi_x_start = 0;     // mm
    int roi_x_end = 0;       // mm
};

struct RackShelfRange {
    std::string shelf_label = "";
    float shelf_start = 0;
    float shelf_end = 0;
};


struct CaliROIBox {
    float x_min, x_max;
    float y_min, y_max;
    float z_min, z_max;
};
CaliROIBox ground_roi_box = {
    -4.0f, -2.0f,
    -0.85f, -0.35f,
    -0.4f, -0.2f
    //0.2f, 0.6f // 포크 사이
};


struct PalletROIBox {
    float x_min, x_max;
    float y_min, y_max;
    float z_min, z_max;
};
PalletROIBox pallet_roi_box = {
    -2.27f, 5.59f,
    -0.7f, -0.35f,
    0.1f, 0.85f
};


struct BackrestROIBox {
    float x_min, x_max;
    float y_min, y_max;
    float z_min, z_max;
};
BackrestROIBox backrest_roi_box = {
    -0.65f, 5.59f,
    -0.3f, 0.0f,
    -0.1f, 0.2f
};



struct WATAConfig {
    int iteration = 0;

    int roi_y_start = 0;  // mm
    int roi_y_end = 0;    // mm
    int roi_z_start = 0;  // mm
    int roi_z_end = 0;    // mm
    int angle = 0;        // deg

    int V_x_start = 0;
    int V_x_end = 0;

    bool read_file = false;
    bool save_file = false;
    std::string read_file_name = "";
    std::string save_file_name = "";
    // 추가
    int mean_k = 0;
    float threshold = 0;
    int height_threshold = 0;
};

struct PalletInfo {
    bool is_pallet;
    float P_height;
    float P_y;
    float P_z;
};

struct BackrestInfo {
    bool is_pallet;
    float P_height;
    float P_y;
    float P_z;
};

// ----------------------------------------------------------------------------
// 전역
// ----------------------------------------------------------------------------
std::mutex control_mutex;
std::mutex g_mutex;
bool V_start_process = false;
bool reading_active = true; // 초기 상태는 읽기 활성화
bool is_paused = false; // 초기 상태는 일시정지 아님

bool pallet_height_fixed = false;
float fixed_pallet_height = 0.0f;

bool heightCalibration_mode = false; // 높이 캘리브레이션 
float fixed_ground_height = 2.272f;
bool ground_height_fixed = false;
bool showGroundROI = false;    // 지면 캘리브레이션용 ROI 박스를 표시할지 여부

bool inFrontofRack = false;

bool PickUp = false;
bool PickUp_1 = false;

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
std::vector<RackShelfRange> g_rack_shelf_ranges;
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
    }
    catch (const nlohmann::json::parse_error& e) {
        std::cerr << "JSON parse error at byte " << e.byte << ": " << e.what() << std::endl;
        throw std::runtime_error("JSON parse error: " + std::string(e.what()));
    }

    cfg.iteration = j.value("iteration", 0);

    if (j.contains("roi")) {
        std::cout << "[DEBUG] Found roi in JSON." << std::endl;

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
        if (roi.contains("rack_shelf_ranges")) {
            for (const auto& shelf : roi["rack_shelf_ranges"]) {
                RackShelfRange rs;
                rs.shelf_label = shelf.value("label", "");
                rs.shelf_start = shelf.value("start", 0);
                rs.shelf_end = shelf.value("end", 0);
                g_rack_shelf_ranges.push_back(rs);
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
        if (pt.x >= ground_roi_box.x_min && pt.x <= ground_roi_box.x_max &&
            pt.y >= ground_roi_box.y_min && pt.y <= ground_roi_box.y_max &&
            pt.z >= ground_roi_box.z_min && pt.z <= ground_roi_box.z_max)
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
            ground_roi_box.x_min, ground_roi_box.x_max,    // X 범위
            ground_roi_box.y_min, ground_roi_box.y_max,  // Y 범위
            ground_roi_box.z_min, ground_roi_box.z_max,     // Z 범위
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
        else if (event.getKeySym() == "v" || event.getKeySym() == "V" || V_start_process) { // 'v' 키: V_start_process 토글
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

// ----------------------------------------------------------------------------
// 고정된 파레트 높이가 어느 스테이지 ROI 범위 안에 있는지 확인
// ----------------------------------------------------------------------------
std::string findPalletStage(bool pallet_height_fixed, float fixed_pallet_height, float current_pallet_height, const std::vector<StageROI>& g_stages)
{
    float pallet_height = pallet_height_fixed ? fixed_pallet_height : current_pallet_height;


    for (const auto& stage : g_stages) {
        float smin = std::min(stage.roi_x_start, stage.roi_x_end) * 0.001f; // mm -> m
        float smax = std::max(stage.roi_x_start, stage.roi_x_end) * 0.001f; // mm->m

        if (pallet_height >= smin - 0.065f && pallet_height <= smax + 0.065f) {
            return stage.label;  // 해당 스테이지 반환
        }
    }
    return "Unknown";  // 어느 스테이지에도 속하지 않으면 "Unknown" 반환
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
            seg.setDistanceThreshold(0.02);
            seg.setInputCloud(cloud);
            seg.segment(*inliers, *coefficients);

            if (inliers->indices.size() < 10) {
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
void calculateAnglePoints(const pcl::PointXYZ& start_point,
    const pcl::PointXYZ& end_point,
    const pcl::PointXYZ& start_x_point,
    const pcl::PointXYZ& end_x_point,
    pcl::visualization::PCLVisualizer::Ptr viewer)
{
    float dy = end_point.y - start_point.y;
    float dx = end_x_point.x - start_x_point.x;
    float angle_radians = std::atan2(dy, dx);  // 기울기 (atan 대신 atan2 사용)

    angle_degrees = angle_radians * 180.0 / M_PI;

    if (angle_degrees < -45.0f) {
        angle_degrees = -45.0f;
    }
    if (angle_degrees > 45.0f) {
        angle_degrees = 45.0f;
    }


    std::cout << "start_max_y_point: (" << start_point.x << ", " << start_point.y << ", " << start_point.z << ")" << std::endl;
    std::cout << "end_max_y_point: (" << end_point.x << ", " << end_point.y << ", " << end_point.z << ")" << std::endl;
    std::cout << "start_max_x_point: (" << start_x_point.x << ", " << start_x_point.y << ", " << start_x_point.z << ")" << std::endl;
    std::cout << "end_max_x_point: (" << end_x_point.x << ", " << end_x_point.y << ", " << end_x_point.z << ")" << std::endl;
    std::cout << "Calculated Angle (degrees): " << angle_degrees << "°" << std::endl;

    viewer->removeShape("angle_line");
    pcl::PointXYZ start_z(start_point.x, start_point.y, start_point.z);
    pcl::PointXYZ end_z(start_point.x, end_point.y, end_point.z);
    viewer->addLine(start_z, end_z, 0.0, 1.0, 1.0, "angle_line");
}


// ----------------------------------------------------------------------------
// 파레트, 백레스트 ROI 박스 표시/제거 함수
// ----------------------------------------------------------------------------
void showPalletROIBox(pcl::visualization::PCLVisualizer::Ptr viewer, bool show) {
    if (show) {
        viewer->removeShape("pallet_roi_box");
        viewer->addCube(
            pallet_roi_box.x_min, pallet_roi_box.x_max,
            pallet_roi_box.y_min, pallet_roi_box.y_max,
            pallet_roi_box.z_min, pallet_roi_box.z_max,
            0.0, 1.0, 0.0, // 녹색 (R=0.0, G=1.0, B=0.0)
            "pallet_roi_box"
        );
        viewer->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
            "pallet_roi_box"
        );
    }
    else {
        viewer->removeShape("pallet_roi_box");
    }
}

void showBackrestROIBox(pcl::visualization::PCLVisualizer::Ptr viewer, bool show) {
    if (show) {
        viewer->removeShape("backrest_roi_box");
        viewer->addCube(
            backrest_roi_box.x_min, backrest_roi_box.x_max,
            backrest_roi_box.y_min, backrest_roi_box.y_max,
            backrest_roi_box.z_min, backrest_roi_box.z_max,
            0.5, 1.0, 0.0, // 녹색 (R=0.0, G=1.0, B=0.0)
            "backrest_roi_box"
        );
        viewer->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
            "backrest_roi_box"
        );
    }
    else {
        viewer->removeShape("backrest_roi_box");
    }
}

// ----------------------------------------------------------------------------
// [높이 센서] 적재물 식별 및 높이 계산 함수
// ----------------------------------------------------------------------------
PalletInfo identifyPallet(pcl::visualization::PCLVisualizer::Ptr viewer,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::vector<int>& shelf_point_counts) {

    PalletInfo info;
    info.is_pallet = false;
    info.P_height = 0.0f;
    info.P_y = 0.0f;
    info.P_z = 0.0f;

    int count_pallet_roi = 0;
    int count_load_roi_1 = 0;
    int count_load_roi_2 = 0;
    int count_load_roi_3 = 0;
    int count_hidden_load = 0;
    float min_x_pallet_roi = std::numeric_limits<float>::max();
    float pallet_y = 0.0f;
    float pallet_z = 0.0f;

    float loadROI_x1_min = pallet_roi_box.x_min + 0.1f;
    float loadROI_x1_max = pallet_roi_box.x_min + 1.6f;
    float loadROI_x2_min = pallet_roi_box.x_min + 2.01f;
    float loadROI_x2_max = pallet_roi_box.x_min + 3.79f;
    float loadROI_x3_min = pallet_roi_box.x_min + 3.94f;
    float loadROI_x3_max = pallet_roi_box.x_min + 5.8f;
    float loadROI_y_min = pallet_roi_box.y_min - 0.2f;
    float loadROI_y_max = pallet_roi_box.y_max;
    float loadROI_z_min = pallet_roi_box.z_min + 0.25f;
    float loadROI_z_max = pallet_roi_box.z_max - 0.25f;

    for (const auto& point : cloud->points) {
        if (point.x >= pallet_roi_box.x_min && point.x <= pallet_roi_box.x_max &&
            point.y >= pallet_roi_box.y_min && point.y <= pallet_roi_box.y_max &&
            point.z >= pallet_roi_box.z_min && point.z <= pallet_roi_box.z_max) {
            count_pallet_roi++;
            if (point.x < min_x_pallet_roi) {
                min_x_pallet_roi = point.x;
                pallet_y = point.y;
                pallet_z = point.z;
            }
        }
        else if (point.x >= loadROI_x1_min && point.x <= loadROI_x1_max &&
            point.y >= loadROI_y_min && point.y <= loadROI_y_max &&
            point.z >= loadROI_z_min && point.z <= loadROI_z_max) {
            count_load_roi_1++;
        }
        else if (point.x >= loadROI_x2_min && point.x <= loadROI_x2_max &&
            point.y >= loadROI_y_min && point.y <= loadROI_y_max &&
            point.z >= loadROI_z_min && point.z <= loadROI_z_max) {
            count_load_roi_2++;
        }
        else if (point.x >= loadROI_x3_min && point.x <= loadROI_x3_max &&
            point.y >= loadROI_y_min && point.y <= loadROI_y_max &&
            point.z >= loadROI_z_min && point.z <= loadROI_z_max) {
            count_load_roi_3++;
        }

        else if (point.x >= -0.1f && point.x <= 0.33f &&
            point.y >= -0.63f && point.y <= -0.38f &&
            point.z >= -0.05f && point.z <= 0.2f)
        {
            count_hidden_load++;
        }
    }

    std::cout << "LOAD points 1: " << count_load_roi_1 << std::endl;
    std::cout << "LOAD points 2: " << count_load_roi_2 << std::endl;
    std::cout << "LOAD points 3: " << count_load_roi_3 << std::endl;
    std::cout << "Hidden Lodad Points: " << count_hidden_load << std::endl;

    if (count_load_roi_1 >= 10 && count_load_roi_1 <= 1000) {
        PickUp_1 = true;
        PickUp = true;
        std::cout << "[PICKUP] 1st Floor !!!" << std::endl;
        viewer->removeShape("load_roi_box_1");
        viewer->addCube(
            loadROI_x1_min, loadROI_x1_max,
            loadROI_y_min, loadROI_y_max,
            loadROI_z_min, loadROI_z_max,
            0.0, 0.5, 1.0,
            "load_roi_box_1"
        );
        viewer->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
            "load_roi_box_1"
        );
        if (count_load_roi_1 <= 0) {
            viewer->removeShape("load_roi_box_1");
        }
    }
    else if (count_load_roi_2 >= 25 && count_load_roi_2 <= 1000) {
        PickUp = true;
        std::cout << "[PICKUP] 2nd Floor !!!" << std::endl;

        viewer->removeShape("load_roi_box_1");
        viewer->addCube(
            loadROI_x2_min, loadROI_x2_max,
            loadROI_y_min, loadROI_y_max,
            loadROI_z_min, loadROI_z_max,
            0.0, 0.5, 1.0,
            "load_roi_box_2"
        );
        viewer->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
            "load_roi_box_2"
        );
        if (count_load_roi_2 <= 0) {
            viewer->removeShape("load_roi_box_2");
        }
    }
    else if (count_load_roi_3 >= 40 && count_load_roi_3 <= 2000) {
        PickUp = true;
        std::cout << "[PICKUP] 3rd Floor !!!" << std::endl;

        viewer->removeShape("load_roi_box_3");
        viewer->addCube(
            loadROI_x3_min, loadROI_x3_max,
            loadROI_y_min, loadROI_y_max,
            loadROI_z_min, loadROI_z_max,
            0.0, 0.5, 1.0,
            "load_roi_box_3"
        );
        viewer->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
            "load_roi_box_3"
        );
        if (count_load_roi_3 <= 0) {
            viewer->removeShape("load_roi_box_3");
        }
    }

    else if (count_hidden_load >= 10) {
        PickUp = true;
        viewer->removeShape("hidden_load_roi_box");
        viewer->addCube(
            -0.1f, 0.33f,
            -0.63f, -0.38f,
            -0.05f, 0.2f,
            1.0, 0.0, 1.0, // 보라색
            "hidden_load_roi_box"
        );
        viewer->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
            "hidden_load_roi_box"
        );
        if (count_hidden_load <= 10) { viewer->removeShape("hidden_load_roi_box"); }
    }

    else {
        PickUp = false;
        PickUp_1 = false;
    }


    std::cout << "pallet points: " << count_pallet_roi << std::endl;

    if (count_pallet_roi >= 15 && count_pallet_roi <= 900 && inFrontofRack == false &&
        !(shelf_point_counts[0] > 800 && shelf_point_counts[1] == 0 && shelf_point_counts[2] == 0))
    {
        info.is_pallet = true; // 그만큼 있으면 파레트(포크)
        info.P_height = min_x_pallet_roi; // 파레트 높이: 가장 아래에 있는 포인트
        info.P_y = pallet_y;
        info.P_z = pallet_z;
        showPalletROIBox(viewer, true); // ROI 박스 활성화
        return info;
    }
    showPalletROIBox(viewer, false); // ROI 박스 비활성화

    int count_backrest_roi = 0;
    float max_x_backrest_roi = std::numeric_limits<float>::min();
    float backrest_y = 0.0f;
    float backrest_z = 0.0f;

    for (const auto& point : cloud->points) {
        if (point.x >= backrest_roi_box.x_min && point.x <= backrest_roi_box.x_max &&
            point.y >= backrest_roi_box.y_min && point.y <= backrest_roi_box.y_max && // 전방 0.3m~0m 사이
            point.z >= backrest_roi_box.z_min && point.z <= backrest_roi_box.z_max) { // 왼쪽으로 0.1m~오른쪽으로 0.2m 사이
            //if (point.x >= -0.27f && point.x <= )

            count_backrest_roi++;
            if (point.x < max_x_backrest_roi) {
                max_x_backrest_roi = point.x;
                backrest_y = point.y;
                backrest_z = point.z;
            }
        }
    }
    std::cout << "backrest points: " << count_backrest_roi << std::endl;

    if (count_backrest_roi >= 20 && count_backrest_roi <= 250 && count_pallet_roi <= 45 && inFrontofRack == false) { // 포인트 개수
        info.is_pallet = true; // 그만큼 있으면 파레트(사실상 백레스트)
        info.P_height = max_x_backrest_roi - 0.05f; // 백레스트
        info.P_y = backrest_y;
        info.P_z = backrest_z;
        showBackrestROIBox(viewer, true); // ROI 박스 활성화

        return info;
    }
    showBackrestROIBox(viewer, false); // ROI 박스 비활성화

    return info;
}

// ----------------------------------------------------------------------------
// [높이 센서] 시각화
// ----------------------------------------------------------------------------

void visualizeHeight(pcl::visualization::PCLVisualizer::Ptr viewer, const PalletInfo& pallet_info, int index) {
    if (!pallet_info.is_pallet) return;

    float effectve_height = (pallet_height_fixed) ? fixed_pallet_height : pallet_info.P_height;
    std::string line_id = "pallet_line_" + std::to_string(index);
    viewer->removeShape(line_id);

    if (ground_height_fixed) {}
    pcl::PointXYZ start_pallet(-fixed_ground_height, pallet_info.P_y, pallet_info.P_z);
    pcl::PointXYZ end_pallet(effectve_height, pallet_info.P_y, pallet_info.P_z);

    viewer->addLine(start_pallet, end_pallet, 0.0, 0.0, 1.0, line_id);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, line_id);

    pallet_line_ids.push_back(line_id);

    std::string text_id = "pallet_height_text_" + std::to_string(index);
    viewer->removeShape(text_id);

    float Pallet_height = effectve_height + fixed_ground_height;
    std::stringstream ss_height;
    ss_height << "Pallet Height: " << Pallet_height << " m";
    std::cout << "Pallet Height: " << Pallet_height << " m\n";

    viewer->addText(ss_height.str(), 20, 170, 20, 1, 1, 1, text_id);
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
    int& previous_index,
    const std::vector<int>& shelf_point_counts) {
    PalletInfo pallet_info = identifyPallet(viewer, cloud_merge, shelf_point_counts);
    float Pallet_height = pallet_info.P_height - fixed_ground_height;

    if (pallet_info.is_pallet) {
        //std::cout << "Pallet Height: " << Pallet_height << " mm" << std::endl;
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

    std::cout << "[DEBUG] g_rack_shelf_ranges size: " << g_rack_shelf_ranges.size() << std::endl;
    for (const auto& shelf : g_rack_shelf_ranges) {
        std::cout << "[DEBUG] Shelf Label: " << shelf.shelf_label
            << ", Start: " << shelf.shelf_start
            << ", End: " << shelf.shelf_end << std::endl;
    }

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

    for (size_t i = 0; i < g_rack_shelf_ranges.size(); ++i) {
        double x1 = (double)std::min(g_rack_shelf_ranges[i].shelf_start, g_rack_shelf_ranges[i].shelf_end) * 0.001;
        double x2 = (double)std::max(g_rack_shelf_ranges[i].shelf_start, g_rack_shelf_ranges[i].shelf_end) * 0.001;
        double y1 = (double)y_min;
        double y2 = (double)y_max;
        double z1 = (double)z_min;
        double z2 = (double)z_max;
        double R = 1.0, G = 0.0, B = 0.0; // 빨강
        std::string shape_id = "rack_shelf_box_" + std::to_string(i);
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
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
            }

            if (READ_PCD_FROM_FILE) { // 파일에서 PCD 로드
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

            if (heightCalibration_mode) { // 지면 캘리브레이션 모드                
                {
                    std::lock_guard<std::mutex> lock(g_mutex);
                    cloud_ground->clear();

                    pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud;
                    if (READ_PCD_FROM_FILE) {
                        src_cloud = cloud_merge;
                    }
                    else {
                        src_cloud = cloud_raw;
                    }

                    for (const auto& pt : src_cloud->points) {
                        if (pt.x >= ground_roi_box.x_min && pt.x <= ground_roi_box.x_max &&
                            pt.y >= ground_roi_box.y_min && pt.y <= ground_roi_box.y_max &&
                            pt.z >= ground_roi_box.z_min && pt.z <= ground_roi_box.z_max)
                        {
                            cloud_ground->push_back(pt);
                        }
                    }
                    cloud_ground->width = static_cast<uint32_t>(cloud_ground->points.size());
                    cloud_ground->height = 1;

                    voxelizePointCloud(cloud_ground, 0.1f, 0.1f, 0.1f);
                }

                try { // 지면 높이 계산
                    fixed_ground_height = std::abs(calculateGroundHeight(cloud_ground));
                    ground_height_fixed = true; // 지면 높이 고정(토글)

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

                    viewer->removePointCloud("roi_cloud");
                    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
                        color_handler(cloud_ground, 255, 0, 0);
                    viewer->addPointCloud<pcl::PointXYZ>(cloud_ground, color_handler, "roi_cloud");
                    viewer->setPointCloudRenderingProperties(
                        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "roi_cloud"
                    ); // 빨강색으로 지면 포인트 표시
                }
                catch (const std::runtime_error& e) {
                    std::cerr << "[ERROR] " << e.what() << std::endl;
                    viewer->removeShape("ground_result_text");
                    viewer->addText("Ground Calibration Failed (No points in ROI)",
                        20, 460, 20, 1.0, 0.0, 0.0,
                        "ground_result_text");
                }
            }

            // ------------------------------------------------------------------
            // [ZMQ] 메시지 수신
            // ------------------------------------------------------------------
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
            // 부피 측정 모드와 아닐 때, 이전 데이터 제거
            // ------------------------------------------------------------------
            if (V_start_process != previous_V_start_process) {
                if (previous_V_start_process) {
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
                    viewer->removePointCloud("filtered_cloud");
                    viewer->removeShape("stageText");
                    for (size_t i = 0; i < g_stages.size(); ++i) {
                        std::string count_text_id = "stage" + std::to_string(i + 1) + "_count";
                        viewer->removeShape(count_text_id);
                    }
                    for (const auto& box_id : cluster_box_ids) {
                        viewer->removeShape(box_id);
                    }
                    viewer->removeAllPointClouds();
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
                // [모드1] 부피 형상 측정
                // -------------------------------------------------------------------------------------
                if (V_start_process && PickUp_1) {
                    // V_start_process가 true인 경우: x < 0.0f인 포인트 처리
                    float start_max_y = std::numeric_limits<float>::lowest();
                    float start_max_x = std::numeric_limits<float>::lowest();
                    pcl::PointXYZ start_max_y_point;
                    pcl::PointXYZ start_max_x_point;
                    float end_max_y = std::numeric_limits<float>::lowest();
                    float end_max_x = std::numeric_limits<float>::lowest();
                    pcl::PointXYZ end_max_y_point;
                    pcl::PointXYZ end_max_x_point;
                    std::vector<float> x_values;

                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcd_local(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_volume(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_angle_filtered(new pcl::PointCloud<pcl::PointXYZ>);

                    for (auto& temp : cloud_merge->points) {
                        pcl::PointXYZ point;
                        point.x = temp.x;
                        point.y = temp.y * COS_THETA - temp.z * SIN_THETA;
                        point.z = temp.y * SIN_THETA + temp.z * COS_THETA;

                        if (point.x < V_x_max && point.x >= -fixed_ground_height + 0.15f &&
                            point.y >= y_min && point.y <= -0.4f &&
                            point.z >= z_min && point.z <= z_max) {
                            x_values.push_back(point.x);
                            cloud_filtered_volume->points.push_back(point);
                            cloud_pcd_local->points.push_back(point);

                            if (point.z >= 0.1f && point.z <= 0.3f) {
                                // X좌표가 가장 큰 점 찾기
                                if (point.x > start_max_x) {
                                    start_max_x = point.x;
                                    start_max_x_point = point;
                                }
                            }

                            if (point.z > 0.7f && point.z <= 0.9f) {
                                // X좌표가 가장 큰 점 찾기
                                if (point.x > end_max_x) {
                                    end_max_x = point.x;
                                    end_max_x_point = point;
                                }
                            }
                        }

                    }
                    for (const auto& point : cloud_merge->points) {
                        if (point.z >= 0.1f && point.z <= 0.3f && point.x == start_max_x) {
                            if (point.y > start_max_y) {
                                start_max_y = point.y;
                                start_max_y_point = point;
                            }
                        }
                        if (point.z > 0.7f && point.z <= 0.9f && point.x == end_max_x) {
                            if (point.y > end_max_y) {
                                end_max_y = point.y;
                                end_max_y_point = point;
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
                    //// Y좌표가 가장 큰 점 찾기
                    //if (point.y > start_max_y) {
                    //    start_max_y = point.y;
                    //    start_max_y_point = point;
                    //}
                    //// Y좌표가 가장 큰 점 찾기
                    //if (point.y > end_max_y) {
                    //    end_max_y = point.y;
                    //    end_max_y_point = point;
                    //}
                    // Voxel Downsample
                    voxelizePointCloud(cloud_filtered_volume, 0.03f, 0.03f, 0.03f);
                    // Outlier Remove
                    removeOutliers(cloud_filtered_volume, config);

                    // Angle Points 계산

                    calculateAnglePoints(start_max_y_point, end_max_y_point, start_max_x_point, end_max_x_point, viewer);

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
                            result_height += (ground_correction_mm + 9.5f);
                        }
                        else {
                            result_height += config.height_threshold + 9.5f;  // 기존 임시 보정 (2755)
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
                        viewer->addText(result.c_str(), 530, 70, 20, 1, 1, 1, "result");

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

                // -------------------------------------------------------------------------------------
                // [모드2] 높이 측정 & 적재 작업 층 분류 + 픽업드롭
                // -------------------------------------------------------------------------------------
                else {
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::PointCloud<pcl::PointXYZ>::Ptr rackShelfCloud(new pcl::PointCloud<pcl::PointXYZ>);

                    std::vector<int> shelf_point_counts(g_rack_shelf_ranges.size(), 0);

                    for (auto& pt : cloud_merge->points) {
                        float px = pt.x;
                        float py = pt.y;
                        float pz = pt.z;

                        float x_min_all = ground_height_fixed ? -fixed_ground_height + 0.005f : -2.272f;
                        float x_max_all = 5.61f;

                        if (px >= x_min_all && px <= x_max_all && py >= y_min && py <= y_max && pz >= z_min && pz <= z_max) {
                            cloud_filtered->push_back(pt);

                            for (size_t i = 0; i < g_rack_shelf_ranges.size(); ++i) {
                                float shelf_start_m = g_rack_shelf_ranges[i].shelf_start / 1000.0f;  // mm -> m 변환
                                float shelf_end_m = g_rack_shelf_ranges[i].shelf_end / 1000.0f;      // mm -> m 변환
                                if (px >= shelf_start_m && px <= shelf_end_m) {
                                    rackShelfCloud->push_back(pt);
                                    shelf_point_counts[i]++;
                                }
                            }
                        }
                    }

                    cloud_filtered->width = cloud_filtered->size();
                    cloud_filtered->height = 1;
                    rackShelfCloud->width = rackShelfCloud->size();
                    rackShelfCloud->height = 1;

                    //std::cout << "[DEBUG] rackShelfCloud points: " << rackShelfCloud->size() << std::endl;

                    // Voxel Downsample
                    voxelizePointCloud(cloud_filtered, 0.05f, 0.05f, 0.05f);

                    // Outlier Remove
                    removeOutliers(cloud_filtered, config);

                    removePreviousPalletVisualizations(viewer);

                    for (size_t i = 0; i < g_rack_shelf_ranges.size(); ++i) {
                        std::string text_id = "shelf_" + std::to_string(i) + "_count";
                        viewer->removeShape(text_id);

                        std::stringstream ss;
                        ss << g_rack_shelf_ranges[i].shelf_label << ": " << shelf_point_counts[i] << " points";

                        viewer->addText(ss.str(), 20, 90 + 20 * i, 20, 1, 0, 0, text_id);
                    }

                    std::cout << "Shelf point 1: " << shelf_point_counts[0] << std::endl;
                    std::cout << "Shelf point 2: " << shelf_point_counts[1] << std::endl;
                    std::cout << "Shelf point 3: " << shelf_point_counts[2] << std::endl;

                    PalletInfo pi = identifyPallet(viewer, cloud_filtered, shelf_point_counts);
                    float Pallet_height = (pallet_height_fixed) ? fixed_pallet_height + fixed_ground_height : pi.P_height + fixed_ground_height;

                    if (shelf_point_counts[0] >= 1800 && shelf_point_counts[1] >= 500 && shelf_point_counts[2] >= 80) {
                        if (!pallet_height_fixed) {
                            fixed_pallet_height = pi.P_height;
                            pallet_height_fixed = true;
                            inFrontofRack = true;
                            std::cout << "[INFO] Fixed Pallet Height: " << Pallet_height << " m" << std::endl;
                        }
                    }
                    else if (Pallet_height <= 2.1f && Pallet_height >= 2.0f &&
                        shelf_point_counts[0] <= 2500 && shelf_point_counts[0] >= 330) {
                        if (!pallet_height_fixed) {
                            fixed_pallet_height = pi.P_height;
                            pallet_height_fixed = true;
                            inFrontofRack = true;
                            std::cout << "[INFO] In Front of Shelf 1, Covered by backrest" << std::endl;
                        }
                    } // 2층 백레스트로 가려져서 포크 높이가 2~2.1m일때, 350~2500개의 선반1~2 점 개수 세고 -> 선반앞, 고정





                    else {
                        if (pallet_height_fixed) {
                            pallet_height_fixed = false;
                            inFrontofRack = false;
                            std::cout << "[INFO] Pallet Height Fixed Reset" << std::endl;
                        }
                    }

                    PalletInfo pallet_info;
                    if (pallet_height_fixed) {
                        pallet_info.is_pallet = true;
                        pallet_info.P_height = fixed_pallet_height;
                    }
                    else {
                        pallet_info = identifyPallet(viewer, cloud_filtered, shelf_point_counts);
                    }

                    static int previous_index = 0;
                    if (pallet_info.is_pallet) {
                        visualizeHeight(viewer, pallet_info, previous_index);
                        previous_index++;
                    }
                    processPoints(cloud_filtered, viewer, previous_pallet_index, shelf_point_counts);

                    std::cout << "LOAD: " << PickUp << std::endl;
                    if (shelf_point_counts[0] >= 1000 && shelf_point_counts[1] == 0 && shelf_point_counts[2] == 0 &&
                        PickUp) {
                        pallet_height_fixed = false;
                        inFrontofRack = false;
                        std::cout << "[INFO] Lifting LOAD between 1~2 stage & Pick Up " << std::endl;
                    }

                    viewer->removeShape("pickup_status");
                    std::string pickup_text = PickUp || PickUp_1 ? "LOAD: PickUp" : "LOAD: Drop";
                    viewer->addText(pickup_text, 20, 320, 30, 0, 1, 0, "pickup_status");

                    //// 클러스터링 및 트래킹 (V_start_process == false)
                    //std::future<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> future_clusters =
                    //    std::async(std::launch::async, performClustering, cloud_filtered);

                    //// 클러스터링 수행
                    //auto P_clusters = future_clusters.get();
                    //// 트래커 업데이트
                    //tracker.update(P_clusters);
                    //// 시각화
                    //visualizeTrackedClusters(viewer, tracker.getTrackedClusters());

                    std::string pallet_stage = findPalletStage(pallet_height_fixed, fixed_pallet_height, pallet_info.P_height, g_stages);
                    std::string fix_status = pallet_height_fixed ? "In front of RACK [ Pallet Height Fixed! ]" : "Normal Status";

                    viewer->removeShape("pallet_height_text");
                    std::stringstream sss_height;
                    sss_height << "Pallet Stage: " << pallet_stage
                        << "\nFolkLift Status: " << fix_status;
                    viewer->addText(sss_height.str(), 20, 270, 25, 0, 1, 0, "pallet_height_text");

                    //// Stage 판별 (포인트 카운팅 방식 사용)
                    //std::string current_stage = "Unknown";
                    //std::vector<int> stage_counts(g_stages.size(), 0);
                    //if (!cloud_filtered->empty()) {
                    //    current_stage = findStageLabel(cloud_filtered, stage_counts);
                    //}

                    viewer->removePointCloud("filtered_cloud");
                    viewer->addPointCloud<pcl::PointXYZ>(cloud_filtered, "filtered_cloud");

                    /*                  viewer->removeShape("stageText");
                                      viewer->addText("Current Stage: " + current_stage,
                                          20, 40, 20, 1, 1, 1, "stageText");*/

                                          //// 텍스트 표시: Stage별 포인트 개수
                                          //for (size_t i = 0; i < g_stages.size(); ++i) {
                                          //    std::string count_text_id = "stage" + std::to_string(i + 1) + "_count";
                                          //    viewer->removeShape(count_text_id);
                                          //    viewer->addText("Stage " + std::to_string(i + 1) + " Points: " + std::to_string(stage_counts[i]),
                                          //        20, 60 + static_cast<int>(i) * 20, // y 위치: 60, 80, 100, 120
                                          //        20, // 글자 크기
                                          //        1, 1, 1, // 흰색
                                          //        count_text_id);
                                          //}


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
