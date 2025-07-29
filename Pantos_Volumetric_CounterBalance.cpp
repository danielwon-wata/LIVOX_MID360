﻿//
// The MIT License (MIT)
//

#define CURL_STATICLIB
#pragma comment(lib, "libcurld.lib")
#pragma comment(lib, "wldap32.lib")
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "Crypt32.lib")

#include "livox_lidar_def.h"
#include "livox_lidar_api.h"
#include "zmq.hpp"

#ifdef _WIN32
#include <winsock2.h>
#else
#include <unistd.h>
#include <arpa/inet.h>
#endif

#include <windows.h>

#include "json.hpp" 
#include <regex>

#include <curl/curl.h>

#include <iomanip>
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
#include <pcl/common/transforms.h>

#include <vtkOutputWindow.h>
#include <vtkObject.h>

#include <unordered_map>
#include <Eigen/Dense>

//#include "resource_Pantos.h"





// 마지막으로 LiDAR 데이터를 받은 시간 (밀리초)
std::atomic<long long> lastLidarTimeMillis(0);

std::atomic<bool> heartbeatRunning(true);

// ----------------------------------------------------------------------------
// JSON 구조체
// ----------------------------------------------------------------------------
struct CaliROIBox {
    float x_min, x_max;
    float y_min, y_max;
    float z_min, z_max;
};
CaliROIBox ground_roi_box = {
    -2.5f, -1.8f,
    -0.65f, -0.3f,
    0.25f, 0.55f
    //0.2f, 0.6f // 포크 사이
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
bool ground_height_fixed = false;
bool showGroundROI = false;    // 지면 캘리브레이션용 ROI 박스를 표시할지 여부

bool inFrontofRack = false;

bool PickUp = false;
bool PickUp_1 = false;

bool READ_PCD_FROM_FILE = false;

int vector_size = 8;

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcd(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_loaded(new pcl::PointCloud<pcl::PointXYZ>);

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_merge(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_raw(new pcl::PointCloud<pcl::PointXYZ>);

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr Cali_cloud_ground(new pcl::PointCloud<pcl::PointXYZ>);

auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>("Pantos_Volumetric_Viewer");

std::vector<float> x_lengths(vector_size); // 고정 크기로 초기화 시킴


std::vector<std::string> volume_line_ids;

float angle_degrees = 0;
int x_index = 0;
int point_index = 0;
int pallet_line_index = 0;

float max_x_value = 0;
float result_length = 0;
float result_height = 0;
float result_width = 0;

static std::atomic<float> imu_roll_deg{ 0.0f };
static std::atomic<float> imu_pitch_deg{ 0.0f };
static std::atomic<long long> last_imu_time_ns{ 0 };
static std::atomic<float>      yaw_rad_accum{ 0.0f };
static std::atomic<float>      imu_yaw_deg{ 0.0f };

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

    auto nowMillis = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    lastLidarTimeMillis.store(nowMillis);

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

    voxel_filter.setDownsampleAllData(true);

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
        else if (event.getKeySym() == "v" || event.getKeySym() == "V") { // 'v' 키: V_start_process 토글
            std::lock_guard<std::mutex> lock(control_mutex);
            V_start_process = !V_start_process; // V_start_process 상태 토글
            if (V_start_process)
                std::cout << "[INFO] V_start_process set to true.\n";
            else
                std::cout << "[INFO] V_start_process set to false.\n";

            // 포인트 클라우드 데이터 초기화
            //resetPointCloudData();

            viewer->removeShape("v_start_process_text");
            std::string initial_status = "V_start_process: " + std::string(V_start_process ? "True" : "False");
            viewer->addText(initial_status, 20, 350, 20, 1, 1, 1, "v_start_process_text");
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
            seg.setDistanceThreshold(0.05);
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
            ec.setMinClusterSize(30);
            ec.setMaxClusterSize(10000);
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
            }

            extract.setNegative(true);
            extract.filter(*cloud);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[detectPlaneYZ ERROR] Exception: " << e.what() << std::endl;
    }
    if (max_length_x > 0) {
        std::string line_id_x = "longest_line_x";

        viewer->removeShape(line_id_x);
        viewer->addLine(p1_x, p2_x, 1.0, 0.0, 0.0, line_id_x);

        bool is_duplicate = false;
        for (size_t i = 0; i < x_lengths.size(); ++i) {
            if (x_lengths[i] == p2_x.x) {
                is_duplicate = true;
                break;
            }
        }

        if (!is_duplicate) {
            std::cout << "Height" + std::to_string(x_index) + " : " + std::to_string(static_cast<int>(p2_x.x * 1000)) << std::endl;
            saveToFile("Height" + std::to_string(x_index) + " : " + std::to_string(static_cast<int>(p2_x.x * 1000)));
            if (x_lengths.size() < vector_size) {
                x_lengths.push_back(p2_x.x);
            }
            else {
                x_lengths[x_index] = p2_x.x;
            }

            x_index = (x_index + 1) % vector_size;

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

void sendHeartbeatSignal() {
    CURL* curl = curl_easy_init();
    if (curl) {
        // 가디언에서 할당된 포인트
        const char* url = "http://localhost:8081/heartbeat";
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_POST, 1L);

        // 전송할 JSON 문자열
        const char* postData = "{\"status\":\"alive\"}";
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postData);

        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "[ERROR] Hearbeat 전송 실패: " << curl_easy_strerror(res) << std::endl;
        }
        curl_easy_cleanup(curl);
    }
}

static std::atomic<bool> heartbeat_enabled{ true };

void heartbeatThreadFunction() {
    while (heartbeatRunning.load()) {
        if (heartbeat_enabled.load()) {

            auto nowMillis = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            long long lastLidarTime = lastLidarTimeMillis.load();
            long long elapsed = nowMillis - lastLidarTime;

            if (elapsed > 5000) { // 5초 이상 경과하면
                std::cerr << "[ERROR] LiDAR 데이터 수신 중단됨. 프로그램 종료." << std::endl;
                heartbeatRunning.store(false);
                break;
            }
            sendHeartbeatSignal(); // heartbeat 신호 전송 함수 전송
        }
        viewer->removeShape("HB_status");
        viewer->addText(std::string("HeartBeat: ") + (heartbeat_enabled.load() ? "O" : "X"), 20, 320, 14, 1.0, 1.0, 0.0, "HB_status");

        std::this_thread::sleep_for(std::chrono::seconds(1)); // 1초마다 신호 전송
    }
}

// 바이어스 계산용 변수
static bool   bias_computed = false;
static int    bias_sample_cnt = 0;
static float  gyro_z_bias_sum = 0.0f;
static float  gyro_z_bias = 0.0f;

void ImuDataCallback(uint32_t handle,
    uint8_t  dev_type,
    LivoxLidarEthernetPacket* packet,
    void* client_data) {
    using namespace std::chrono;

    // 1) bias 계산 단계: 첫 200샘플(약 1초) 동안 gyro_z 평균값을 bias로 사용
    auto* imu_arr = reinterpret_cast<LivoxLidarImuRawPoint*>(packet->data);
    float gz = imu_arr[packet->dot_num - 1].gyro_z;

    if (!bias_computed) {
        gyro_z_bias_sum += gz;
        if (++bias_sample_cnt >= 200) {
            gyro_z_bias = gyro_z_bias_sum / bias_sample_cnt;
            bias_computed = true;
        }
        return;  // bias 취득 전에는 적분 안 함
    }

    // 2) 시간 간격 계산 (nanosecond 단위)
    auto now_ns = duration_cast<nanoseconds>(
        steady_clock::now().time_since_epoch()
    ).count();
    long long last_ns = last_imu_time_ns.load();
    float dt = 0.0f;
    if (last_ns > 0) {
        dt = (now_ns - last_ns) * 1e-9f;  // [s]
    }
    last_imu_time_ns.store(now_ns);

    // 3) bias 보정 및 노이즈 임계치
    float gz_corr = gz - gyro_z_bias;
    if (std::abs(gz_corr) < 0.005f)  // 0.005 rad/s 이하는 무시
        gz_corr = 0.0f;

    // 4) 적분 → yaw 누적
    float delta = gz_corr * dt;      // [rad]
    float accum = yaw_rad_accum.load() + delta;
    yaw_rad_accum.store(accum);
    imu_yaw_deg.store(accum * 180.0f / M_PI);  // [deg]
}

float estimateGroundYawOffset(
    pcl::PointCloud<pcl::PointXYZ>::Ptr const& cloud_ground)
{
    // 1. RANSAC 평면 피팅
    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.02f);         // 2cm 이내를 평면으로 간주
    seg.setInputCloud(cloud_ground);
    seg.segment(*inliers, *coeff);

    if (inliers->indices.empty()) {
        // 평면 못 찾았을 때는 보정 안함
        return 0.0f;
    }

    // 2. 평면 법선 벡터 (a,b,c)
    Eigen::Vector3f normal(coeff->values[0],
        coeff->values[1],
        coeff->values[2]);
    normal.normalize();

    // 3. x축(1,0,0) 기준으로 z축 회전량 (yaw 오프셋)
    //    atan2(b, a) 을 센서에 역으로 적용해야 지면이 y-z 평면과 평행해짐
    float yaw_offset = std::atan2(normal.y(), normal.x());
    return yaw_offset;
}

// 2) yaw 보정량 만큼 포인트클라우드를 역회전(transform) 시킴
//    src: 원본 클라우드, dst: 변환 후 클라우드, yaw_offset: 위 함수에서 얻은 라디안
void applyYawCalibration(
    pcl::PointCloud<pcl::PointXYZ>::Ptr const& src,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& dst,
    float yaw_offset)
{
    Eigen::Affine3f tf = Eigen::Affine3f::Identity();
    // 지면 법선->x축 보정을 위해 –yaw_offset 만큼 z축 회전
    tf.rotate(Eigen::AngleAxisf(-yaw_offset, Eigen::Vector3f::UnitZ()));
    dst.reset(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*src, *dst, tf);
}


// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, const char* argv[]) {
    // 시스템 오류 대화상자 표시를 막음 (Windows 전용)
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

    vtkOutputWindow::GetInstance()->PromptUserOff();
    vtkObject::GlobalWarningDisplayOff();


    std::cout << "Current working directory: "
        << std::filesystem::current_path() << std::endl;

    curl_global_init(CURL_GLOBAL_ALL); // curl 전역 초기화 (프로그램 시작 시 한번 호출)

    // 프로그램 시작 시, 마지막 LiDAR 수신시간 초기화
    lastLidarTimeMillis.store(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());

    std::thread heartbeatThread(heartbeatThreadFunction);
    heartbeatThread.detach(); // 스레드를 분리하여 독립적으로 실행

    std::vector<float> ground_samples;

    // 1) 설정 로드
    const std::string LIVOX_PATH = "config/config.json";
    const std::string WATA_PATH = "config/P_setting.json";
    WATAConfig config = readConfigFromJson(WATA_PATH);

    bool READ_PCD_FROM_FILE = config.read_file;
    bool SAVE_PCD_FROM_FILE = config.save_file;
    const std::string READ_PCD_FILE_NAME = config.read_file_name;
    const std::string SAVE_PCD_FILE_NAME = config.save_file_name;

    float fixed_ground_height = 2.026;

    const int iteration = config.iteration;

    float V_x_min = std::min(config.V_x_start, config.V_x_end) * 0.001f;
    float V_x_max = std::max(config.V_x_start, config.V_x_end) * 0.001f;

    // 3) y,z ROI => mm->m
    float y_min = std::min(config.roi_y_start, config.roi_y_end) * 0.001f; // ex) -1.6
    float y_max = std::max(config.roi_y_start, config.roi_y_end) * 0.001f; // ex) -0.3
    float z_min = std::min(config.roi_z_start, config.roi_z_end) * 0.001f;
    float z_max = std::max(config.roi_z_start, config.roi_z_end) * 0.001f;


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
    else {  // 실시간 (Livox) 모드
        try {
            if (!LivoxLidarSdkInit(LIVOX_PATH.c_str())) {
                throw std::runtime_error("Livox Init Failed");
            }
            SetLivoxLidarPointCloudCallBack(PointCloudCallback, &config);
            SetLivoxLidarImuDataCallback(ImuDataCallback, &config);
        }
        catch (const std::exception& ex) {
            std::cerr << "[ERROR] Exception during Init: " << ex.what() << std::endl;

            LivoxLidarSdkUninit();
            exit(EXIT_FAILURE);
        }
    }

    // ZMQ
    zmq::context_t context(1);
    zmq::socket_t publisher(context, ZMQ_PUB);
    publisher.bind("tcp://127.0.0.1:5001");

    zmq::socket_t subscriber(context, ZMQ_SUB);
    subscriber.connect("tcp://127.0.0.1:5002");
    subscriber.set(zmq::sockopt::subscribe, "LIS>MID360");

    // 4) Viewer
    viewer->addCoordinateSystem(1.0);
    viewer->setBackgroundColor(0.1, 0.1, 0.1);
    viewer->setCameraPosition(4.14367, 5.29453, -3.91817, 0.946026, -0.261667, 0.191218);

    int previous_pallet_index = 0;

    viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)viewer.get());


    x_lengths.clear();


    // V_start_process 상태 텍스트 초기화
    std::string initial_status = "V_start_process: " + std::string(V_start_process ? "True" : "False");
    viewer->addText(initial_status, 20, 350, 20, 1, 1, 1, "v_start_process_text");


    // 전역 변수 또는 메인 루프 상단에 추가
    bool previous_V_start_process = V_start_process;

    static float last_yaw_mesurement = 0.0f;
    static float cumulative_yaw = 0.0f;
    static bool   stable_timer_running = false;
    static std::chrono::steady_clock::time_point stable_start_time;

    bool firstCalibrationDone = false;    

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
                }
            }

            if (heightCalibration_mode) { // 지면 캘리브레이션 모드                
                {
                    std::lock_guard<std::mutex> lock(g_mutex);
                    cloud_ground->clear();

                    pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud;
                    src_cloud = cloud_merge;

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
                saveToFile("[" + message + "], timestamp: " + getCurrentTime());
            }

            // ------------------------------------------------------------------
            // 부피 측정 모드와 아닐 때, 이전 데이터 제거
            // ------------------------------------------------------------------
            if (V_start_process != previous_V_start_process) {
                if (previous_V_start_process) {
                    //viewer->removePointCloud("cloud_pcd");
                    viewer->removeShape("result");
                    viewer->removeShape("angle_line");
                    for (const auto& line_id : volume_line_ids) {
                        viewer->removeShape(line_id);
                    }
                    volume_line_ids.clear();
                }


                // 포인트 클라우드 데이터 초기화
                resetPointCloudData();

                // 이전 상태 업데이트
                previous_V_start_process = V_start_process;

                // Viewer에 현재 상태 텍스트 업데이트
                std::string status_text = "V_start_process: " + std::string(V_start_process ? "True" : "False");
                viewer->removeShape("v_start_process_text");
                viewer->addText(status_text, 20, 350, 20, 1, 1, 1, "v_start_process_text");

                std::cout << "[INFO] V_start_process state changed to "
                    << (V_start_process ? "True" : "False") << ". Data has been reset." << std::endl;
            }


            Cali_cloud_ground->clear();
            pcl::PointCloud<pcl::PointXYZ>::Ptr cali_cloud;
            cali_cloud = cloud_merge;

            for (const auto& pt : cali_cloud->points) {
                if (pt.x >= ground_roi_box.x_min && pt.x <= ground_roi_box.x_max &&
                    pt.y >= ground_roi_box.y_min - 0.35f && pt.y <= ground_roi_box.y_max &&
                    pt.z >= ground_roi_box.z_min && pt.z <= ground_roi_box.z_max)
                {
                    Cali_cloud_ground->push_back(pt);
                }
            }
            Cali_cloud_ground->width = static_cast<uint32_t>(Cali_cloud_ground->points.size());
            Cali_cloud_ground->height = 1;

            voxelizePointCloud(Cali_cloud_ground, 0.1f, 0.1f, 0.1f);                

            // ------------------------------------------------------------------
            // 1층 픽업

            //std::lock_guard<std::mutex> lk(g_mutex);

            int count_load_roi_1 = 0;
            float loadROI_x1_min = -fixed_ground_height + 0.05f;
            float loadROI_x1_max = -fixed_ground_height + 2.0f;
            float loadROI_y_min = -1.1f;
            float loadROI_y_max = -0.18f;
            float loadROI_z_min = 0.25f;
            float loadROI_z_max = 0.55f;

            pcl::PointCloud<pcl::PointXYZ>::Ptr Pickup_cloud;
            Pickup_cloud = cloud_merge;

            for (const auto& point : Pickup_cloud->points) {
                if (point.x >= loadROI_x1_min && point.x <= loadROI_x1_max &&
                    point.y >= loadROI_y_min && point.y <= loadROI_y_max &&
                    point.z >= loadROI_z_min && point.z <= loadROI_z_max) {
                    count_load_roi_1++;
                }
            }

            if (count_load_roi_1 >= 10 && count_load_roi_1 <= 5000) {
                PickUp_1 = true;
                heightCalibration_mode = false;

                std::ostringstream pickOSS;
                pickOSS << "1st PickUp!!\n" << "pts: " << count_load_roi_1;
                viewer->removeShape("pickup_text");
                viewer->addText(pickOSS.str(), 20, 200, 14, 1.0, 1.0, 0.0, "pickup_text");

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
            }
            else
            {
                heightCalibration_mode = true;
                viewer->removeShape("load_roi_box_1");
                viewer->removeShape("pickup_text");
            }


            // 프로그램 재시동 및 처음 켰을 때,
            // 파레트 보이면 파레트 안보이게 하라고 HB안보냄
            if (!firstCalibrationDone) { // 초기세팅 이전
                if (!PickUp_1) { // 지면이 보이는 상황 (파레트 안보임)
                    float yaw_offset = estimateGroundYawOffset(cloud_ground);
                    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>());
                    applyYawCalibration(cloud_merge, tmp, yaw_offset);
                    *cloud_merge = *tmp;

                    fixed_ground_height = std::abs(calculateGroundHeight(cloud_ground));

                    std::ostringstream YawCaliOss;
                    YawCaliOss << "[INIT] Ground Cali: yaw_offset="
                        << yaw_offset << "rad, height="
                        << fixed_ground_height << "m";
                    viewer->removeShape("YawCali_text");
                    viewer->addText(YawCaliOss.str(), 250, 20, 14, 0.0, 1.0, 0.0, "YawCali_text");

                    heartbeat_enabled.store(true); // HB 보냄
                    firstCalibrationDone = true; // 초기세팅 완료
                }
                else { // 지면은 안보이고 파레트가 보이는 상황
                    heartbeat_enabled.store(false); // HB 안보냄
                    // firstCaliDone을 false로 유지하기 때문에
                    // HB는 안보내서 LIS는 프로그램이 꺼진걸로 판단(측정불가 상황)
                    // 재시동 하더라도 파레트만 보인다면 계속 루프
                    // 지면이 보이게 되면 firstCaliDone true되면서 빠져나옴
                }
            }

            viewer->removeShape("firstCaliDone");
            viewer->addText(
                std::string("FirstCalibration Done: ") +
                (firstCalibrationDone ? "Done" : "Undone\n(Clear the Palette!)"), 
                20, 140, 14, 1.0, 1.0, 0.0, "firstCaliDone");

            // 초기 세팅 = 지면을 라이다 좌표계와 수평으로 맞춤
            // 이후 = 변해지는 각도값만큼 회전 조절, 기존 부피측정 로직으로 수행



            float current_yaw_deg = imu_yaw_deg.load();
            float delta_yaw = current_yaw_deg - last_yaw_mesurement;

            if (std::abs(delta_yaw) > 0.1f) {
                cumulative_yaw += delta_yaw;
                last_yaw_mesurement = current_yaw_deg;
                stable_timer_running = false;
                heightCalibration_mode = false;
            }
            else {
                if (!stable_timer_running) {
                    stable_timer_running = true;
                    stable_start_time = std::chrono::steady_clock::now();
                }
                else {
                    auto elapsed = std::chrono::duration<float>(
                        std::chrono::steady_clock::now() - stable_start_time
                    ).count();

                    if (elapsed >= 1.0f) {
                        last_yaw_mesurement = 0.0f;
                        imu_yaw_deg.store(0.0f);
                        yaw_rad_accum.store(0.0f);
                        stable_timer_running = false;
                    }
                }
            }
            float cum_rad = cumulative_yaw * M_PI / 180.0f;
            Eigen::Affine3f tf = Eigen::Affine3f::Identity();
            tf.rotate(Eigen::AngleAxisf(cum_rad, Eigen::Vector3f::UnitZ()));

            pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());

            pcl::transformPointCloud(*cloud_merge, *aligned, tf);


            viewer->removePointCloud("raw");
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rawHandler(aligned, 255, 255, 255);
            viewer->addPointCloud(aligned, rawHandler, "raw");
            viewer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "raw");

            viewer->removeShape("imu_yaw_text");
            std::ostringstream oss;
            oss << "Yaw: " << std::fixed << std::setprecision(1) << current_yaw_deg << " deg\n"
                << "Delta Yaw: " << std::fixed << std::setprecision(1) << delta_yaw << " deg";
            viewer->addText(oss.str(), 20, 70, 15, 1, 1, 0, "imu_yaw_text");

            viewer->removeShape("cur_height");
            std::ostringstream curHeightOss;
            curHeightOss << "Ground Height: " << fixed_ground_height << " m\n"
                << "HeightCalibration: " << (heightCalibration_mode ? "O" : "X");
            viewer->addText(curHeightOss.str(), 300, 400, 14, 1.0, 1.0, 1.0, "cur_height");


            // -------------------------------------------------------------------------------------
            // 부피 형상 측정
            // -------------------------------------------------------------------------------------
            if (V_start_process && PickUp_1) {

                std::lock_guard<std::mutex> lk(g_mutex);


                if (cloud_merge && !cloud_merge->empty()) {

                    std::vector<float> x_values;

                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcd_local(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_volume(new pcl::PointCloud<pcl::PointXYZ>);

                    for (auto& temp : cloud_merge->points) {
                        pcl::PointXYZ point;
                        point.x = temp.x;
                        point.y = temp.y;
                        point.z = temp.z;


                        if (point.x < -fixed_ground_height + 2.18f && point.x >= -fixed_ground_height + 0.18f &&
                            point.y >= y_min && point.y <= -0.15f && // 기본값 -0.45f (백레스트 앞)
                            point.z >= z_min && point.z <= z_max) {
                            x_values.push_back(point.x);
                            cloud_filtered_volume->points.push_back(point);
                            cloud_pcd_local->points.push_back(point);
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
                    voxelizePointCloud(cloud_filtered_volume, 0.05f, 0.02f, 0.05f);
                    // Outlier Remove
                    removeOutliers(cloud_filtered_volume, config);

                    detectPlaneYZ(cloud_filtered_volume, viewer);

                    // 부피 측정 결과 계산
                    if (x_lengths.size() == vector_size) {
                        result_height = calculateAverageX(x_lengths) * 1000;
                        x_lengths.clear();
                    }



                    if (result_height != 0 && result_width == 0 && result_length == 0) {
                        bool result_status = true;

                        float ground_correction_mm = 0.0f;
                        ground_correction_mm = fixed_ground_height * 1000.0f;
                        result_height += ground_correction_mm;

                        std::string json_result = "{"
                            "\"height\": " + std::to_string(result_height) + ", "
                            "\"width\": " + std::to_string(result_width) + ", "
                            "\"length\": " + std::to_string(result_length) + ", "
                            "\"result\": " + std::to_string(result_status) + ", "
                            "\"timestamp\": \"" + getCurrentTime() + "\", "
                            "\"points\": [] }";


                        std::ostringstream oss;
                        oss << "Height: " << result_height << " mm \n"
                            << "Width: " << result_width << " mm \n"
                            << "Length: " << result_length << " mm \n"
                            << "LiDAR Height: " << ground_correction_mm << " mm \n";

                        std::string result = oss.str();

                        viewer->removeShape("result");
                        viewer->addText(result.c_str(), 450, 70, 20, 1, 1, 1, "result");

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
                // PCD 데이터 저장 (단순 누적)
                if (SAVE_PCD_FROM_FILE) {
                    *cloud_pcd += *cloud_merge;
                    std::cout << "[DEBUG] Accumulated " << cloud_pcd->size() << " points in cloud_pcd. \n";
                }

                // 파일 모드라면 다음 chunk 대기 위해 처리 종료
                if (READ_PCD_FROM_FILE) {
                    cloud_merge->clear();
                }

                // 클라우드 처리 후 클라우드_merge 초기화
                //cloud_merge->clear();
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
    curl_global_cleanup(); // curl 전역 정리 (프로그램 종료 시 한번 호출)
    return 0;
}
