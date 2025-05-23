//
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

#include <curl/curl.h>

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

#include <vtkSmartPointer.h>
#include <vtkSliderWidget.h>
#include <vtkSliderRepresentation2D.h>
#include <vtkButtonWidget.h>
#include <vtkTexturedButtonRepresentation2D.h>
#include <vtkPNGReader.h>
#include <vtkCommand.h>
#include <vtkImageData.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkTextWidget.h>
#include <vtkTextRepresentation.h>
#include <vtkOutputWindow.h>
#include <vtkObject.h>
#include <vtkImageSlice.h>
#include <vtkImageSliceMapper.h>


#include "resource_dev.h"










// ----------------------------------------------------------------------------
// JSON 구조체
// ----------------------------------------------------------------------------
struct CaliROIBox {
    float x_min, x_max;
    float y_min, y_max;
    float z_min, z_max;
};
CaliROIBox ground_roi_box = {
    -3.5f, -1.0f,
    -0.85f, -0.35f,
    0.2f, 0.5f
    //0.2f, 0.6f // 포크 사이
};




struct WATAConfig {
    int iteration = 0;

    int mean_k = 0;
    float threshold = 0;
	float reach_height = 0.0f;
	float counterbalance_height = 0.0f;

    bool flag_detect_plane_yz = false;
    bool flag_load_roi = false;
    bool flag_raw_cloud = false;
    bool flag_intensity = false;
    bool flag_dual_emit = false;
    bool flag_reach_off_counter = false;
    bool flag_replay = false;
    bool flag_heart_beat = true;

    bool read_file = false;
    bool save_file = false;
    std::string read_file_name = "";
    std::string save_file_name = "";
};


// ----------------------------------------------------------------------------
// VTK 콜백 클래스들
// ----------------------------------------------------------------------------
struct SliderCallback : public vtkCommand {
    static SliderCallback* New() { return new SliderCallback; }

    WATAConfig* iteration_ptr = nullptr;
	WATAConfig* meank_ptr = nullptr;
	WATAConfig* threshold_ptr = nullptr;

    void Execute(vtkObject* caller, unsigned long, void*) override {
        auto slider = static_cast<vtkSliderWidget*>(caller);
        double v = static_cast<vtkSliderRepresentation*>(slider->GetRepresentation())->GetValue();
        if (iteration_ptr) {
            iteration_ptr->iteration = static_cast<int>(v);
            std::cout << "[SLIDER] iteration = " << iteration_ptr->iteration << std::endl;
        }
        if (meank_ptr) {
			meank_ptr->mean_k = static_cast<int>(v);
			std::cout << "[SLIDER] mean_k = " << meank_ptr->mean_k << std::endl;
        }
        if (threshold_ptr) {
            threshold_ptr->threshold = static_cast<float>(v);
            std::cout << "[SLIDER] threshold = " << threshold_ptr->threshold << std::endl;
        }

    }    
};

struct ButtonCallback : public vtkCommand {
    static ButtonCallback* New() { return new ButtonCallback; }

	bool* toggleFlag = nullptr;
    std::string name;

    void Execute(vtkObject* caller, unsigned long, void*) override {
        if (toggleFlag) {
			*toggleFlag = !*toggleFlag;
			std::cout << "[BUTTON] " << name << " is now " << (*toggleFlag ? "ON" : "OFF") << std::endl;
        }
    }      
};








// ----------------------------------------------------------------------------
// 전역
// ----------------------------------------------------------------------------
std::mutex control_mutex;
std::mutex g_mutex;
std::atomic<long long> lastLidarTimeMillis(0);
std::atomic<bool> heartbeatRunning(true);
static std::atomic<bool> rebootRequested{ false };


bool V_start_process = false;
bool reading_active = true; // 초기 상태는 읽기 활성화
bool is_paused = false; // 초기 상태는 일시정지 아님
float fixed_ground_height = 0.0f;

bool pallet_height_fixed = false;
float fixed_pallet_height = 0.0f;

bool heightCalibration_mode = false; // 높이 캘리브레이션 

bool ground_height_fixed = false;
bool showGroundROI = false;    // 지면 캘리브레이션용 ROI 박스를 표시할지 여부

bool inFrontofRack = false;

bool PickUp = false;
bool PickUp_1 = false;

bool READ_PCD_FROM_FILE = false;

//int iteration = 500;
int vector_size = 8;

pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_pcd(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_loaded(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_merge(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_raw(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZI>);

std::vector<float> x_lengths(vector_size);
std::vector<float> y_lengths(vector_size);
std::vector<float> z_lengths(vector_size);

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

FovCfg fov_cfg0 = {    // yaw_start, yaw_stop, pitch_start(안씀), pitch_stop(안씀)
  0,     // yaw_start = 0°
  360,   // yaw_stop  = 360°
  0,0
};
FovCfg fov_cfg1 = {    // yaw_start(안씀), yaw_stop(안씀), pitch_start, pitch_stop
  0,0,
  -2,    // pitch_start = -2°
  52     // pitch_stop  = +52°
};


std::atomic<uint32_t> g_lidar_handle{ 0 };
std::atomic<bool>   g_dual_emit_set{ false };


bool enableDetectPlaneYZ = false;
bool enableLoadROI = false;
bool enableRAWcloud = false;
bool enableINTENSITY = false;
bool enableDualEmit = false;
bool onReachoffCounter = false;
bool enableReplay = false;
bool enableHeartBeat = true;


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

    if (j.contains("flags")) {
        const auto& f = j["flags"];
        cfg.flag_detect_plane_yz = f.value("detectPlaneYZ", cfg.flag_detect_plane_yz);
        cfg.flag_load_roi = f.value("loadROI", cfg.flag_load_roi);
        cfg.flag_raw_cloud = f.value("rawCloud", cfg.flag_raw_cloud);
        cfg.flag_intensity = f.value("intensity", cfg.flag_intensity);
        cfg.flag_dual_emit = f.value("dualEmit", cfg.flag_dual_emit);
        cfg.flag_reach_off_counter = f.value("reachOffCounter", cfg.flag_reach_off_counter);
        cfg.flag_replay = f.value("replay", cfg.flag_replay);
        cfg.flag_heart_beat = f.value("heartBeat", cfg.flag_heart_beat);
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

	cfg.reach_height = j.value("reach_height", 0.0f);
	cfg.counterbalance_height = j.value("counterbalance_height", 0.0f);

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
    oss << "Camera Pos: ("
        << camera.pos[0] << ", "
        << camera.pos[1] << ", "
        << camera.pos[2] << ", "
        << camera.pos[3] << ", "
        << camera.pos[4] << ", "
        << camera.pos[5] << ")";

    // 기존 텍스트 제거
    viewer->removeShape("camera_position_text");

    // 새로운 텍스트 추가
    viewer->addText(oss.str(), 20, 705, 10, 1.0, 1.0, 1.0, "camera_position_text");
}

void DualEmitCb(livox_status status, uint32_t handle, LivoxLidarAsyncControlResponse * response, void* client_data) {
    if (status == kLivoxLidarStatusSuccess) {
        std::cout << "[DUAL EMIT] Enabled on handle " << handle << "\n";
    }
    else {
        std::cerr << "[DUAL EMIT] Failed: " << status << "\n";
    }

}





// ----------------------------------------------------------------------------
// 리소스 콜백 (실시간 모드)
// ----------------------------------------------------------------------------
void PointCloudCallback(uint32_t handle, const uint8_t dev_type, LivoxLidarEthernetPacket* data, void* client_data) {
    if (g_lidar_handle.load() == 0) {
        g_lidar_handle = handle;

        SetLivoxLidarFovCfg0(handle, &fov_cfg0, nullptr, nullptr);
        SetLivoxLidarFovCfg1(handle, &fov_cfg1, nullptr, nullptr);
        EnableLivoxLidarFov(handle, 1, nullptr, nullptr);
    }
	if (enableDualEmit && !g_dual_emit_set.load()) {
		g_dual_emit_set = true;
		SetLivoxLidarDualEmit(handle, true, DualEmitCb, nullptr);
	}
    if (!data) return;

    auto nowMillis = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    lastLidarTimeMillis.store(nowMillis);

    const WATAConfig* config = static_cast<const WATAConfig*>(client_data);
    if (data->data_type == kLivoxLidarCartesianCoordinateHighData) {
        auto* p_point_data = reinterpret_cast<LivoxLidarCartesianHighRawPoint*>(data->data);
        std::lock_guard<std::mutex> lock(g_mutex);

        for (uint32_t i = 0; i < data->dot_num; i++) {
            pcl::PointXYZI point;
            point.x = p_point_data[i].x / 1000.0f;
            point.y = p_point_data[i].y / 1000.0f;
            point.z = p_point_data[i].z / 1000.0f;
            point.intensity = static_cast<float>(p_point_data[i].reflectivity);

            cloud_raw->points.push_back(point);
        }

        cloud_raw->width = cloud_raw->points.size();
        cloud_raw->height = 1;

        if (cloud_raw->points.size() >= 96 * config->iteration) {
            std::swap(cloud_raw, cloud_merge);
            cloud_raw->clear();
        }
    }
}

// ----------------------------------------------------------------------------
// 전처리 1) Voxel
// ----------------------------------------------------------------------------
void voxelizePointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
    float x_leaf, float y_leaf, float z_leaf)
{
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(x_leaf, y_leaf, z_leaf);

    voxel_filter.setDownsampleAllData(true);

    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>());
    voxel_filter.filter(*filtered);

    cloud->clear();
    *cloud = *filtered;
}

// ----------------------------------------------------------------------------
// 전처리 2) Outlier Remove
// ----------------------------------------------------------------------------
void removeOutliers(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, const WATAConfig& config) {
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
    sor.setInputCloud(cloud);

    sor.setMeanK(config.mean_k);
    sor.setStddevMulThresh(config.threshold);  // 더 커지면 제거가 줄고, 더 작아지면 제거가 많아짐

    sor.filter(*cloud);
}

// ----------------------------------------------------------------------------
// [자동 높이 캘리브레이션] 지면 탐지
// ----------------------------------------------------------------------------
float calculateGroundHeight(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ground) {
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
            viewer->addText(initial_status, 20, 630, 20, 1, 1, 1, "v_start_process_text");
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
void detectPlaneYZ(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, pcl::visualization::PCLVisualizer::Ptr viewer) {
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ExtractIndices<pcl::PointXYZI> extract;

    float max_length_x = 0.0;
    float max_length_y = 0.0;
    float max_length_z = 0.0;
    pcl::PointXYZI p1_x, p2_x;
    pcl::PointXYZI p1_y, p2_y;
    pcl::PointXYZI p1_z, p2_z;

    // 전역 영역의 평면 (반사율 활용 예정)
    pcl::PointCloud<pcl::PointXYZI>::Ptr planeCloud(new pcl::PointCloud<pcl::PointXYZI>());
    try {
        while (true) {
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.05);
			seg.setInputCloud(cloud); // cloud에서 평면을 찾음
			seg.segment(*inliers, *coefficients); // inliers에 평면 인덱스 저장


            if (inliers->indices.size() < 100) {
                std::cout << "No more planes found." << std::endl;
                break;
            }
            std::cout << "[info] plane indices size: " << inliers->indices.size() << std::endl;
            std::cout << "[DEBUG] Input cloud size: " << cloud->size() << std::endl;


            // 평면의 포인트 클라우드 생성
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZI>);
            extract.setInputCloud(cloud);
            extract.setIndices(inliers);
            extract.setNegative(false);
            extract.filter(*cloud_plane);

            // 클러스터링
            pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
            tree->setInputCloud(cloud_plane);

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
            ec.setClusterTolerance(0.06);
            ec.setMinClusterSize(80);
            ec.setMaxClusterSize(3000);
            seg.setMaxIterations(5000);  // 기본값 1000보다 증가
            seg.setProbability(0.99); // 신뢰도를 높임

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
                    p1_x = pcl::PointXYZI(min_x, min_y, min_z);
                    p2_x = pcl::PointXYZI(max_x, min_y, min_z);
                }

                //std::cout << "max_y : " << max_y << " min_y : " << min_y << std::endl;
                float length_y = max_y - min_y; // Y축 방향 길이
                if (length_y > max_length_y) {
                    max_length_y = length_y;
                    p1_y = pcl::PointXYZI(min_x, min_y, min_z);
                    p2_y = pcl::PointXYZI(min_x, max_y, min_z);
                }

                float length_z = max_z - min_z; // Z축 방향 길이
                if (length_z > max_length_z) {
                    max_length_z = length_z;
                    p1_z = pcl::PointXYZI(min_x, min_y, min_z);
                    p2_z = pcl::PointXYZI(min_x, min_y, max_z);
                }
                planeCloud = cloud_plane; // 전체 평면 클라우드 저장 (계산 후 사용)

            }

            extract.setNegative(true); // 이후 클라우드 제거 처리
            extract.filter(*cloud);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[detectPlaneYZ ERROR] Exception: " << e.what() << std::endl;
    }
    if (max_length_y > 0) {
        std::string line_id_y = "longest_line_y";

        viewer->removeShape(line_id_y);
        pcl::PointXYZ start_y(p1_y.x, p1_y.y, p1_y.z);
        pcl::PointXYZ end_y(p1_y.x, p1_y.y + max_length_y, p1_y.z);
        viewer->addLine(start_y, end_y, 0.0, 1.0, 0.0, line_id_y);

        viewer->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
            3, line_id_y);

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

        struct CornerY {
            float x, y, z;
        };

        CornerY A = { p1_y.x, p1_y.y, p2_z.z };
        CornerY B = { p1_y.x, p1_y.y + max_length_y, p2_z.z };

        float delta_y = 0.15f;

        float roi_min_x = A.x - delta_y;
        float roi_max_x = A.x + delta_y;
        // Y는 두 점의 최소/최대값
        float roi_min_y = std::min(A.y, B.y);
        float roi_max_y = std::max(A.y, B.y);
        float roi_min_z = A.z - delta_y;
        float roi_max_z = A.z + delta_y;

        int point_count = 0;
        float sumIntensity = 0.0f;

        if (enableINTENSITY) {
            for (const auto& pt : planeCloud->points) {
                if (pt.x >= roi_min_x && pt.x <= roi_max_x &&
                    pt.y >= roi_min_y && pt.y <= roi_max_y &&
                    pt.z >= roi_min_z && pt.z <= roi_max_z)
                {
                    point_count++;
                    sumIntensity += pt.intensity;
                }
            }

            float averageIntensity = (point_count > 0) ? (sumIntensity / point_count) : 0.0f;

            std::string roiCubeName = "roi_edge_y_box";
            viewer->removeShape(roiCubeName);
            viewer->addCube(roi_min_x, roi_max_x,
                roi_min_y, roi_max_y,
                roi_min_z, roi_max_z,
                1.0, 0.0, 1.0, roiCubeName);
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, roiCubeName);

            std::ostringstream textOss;
            textOss << "Count(Y): " << point_count << " | Intensity: " << averageIntensity;
            std::string roiTextName = "roi_edge_y_text";
            viewer->removeShape(roiTextName);
            viewer->addText3D(textOss.str(),
                pcl::PointXYZ((roi_min_x + roi_max_x) / 2,
                    (roi_min_y + roi_max_y) / 2,
                    (roi_min_z + roi_max_z) / 2),
                0.05, 1.0, 1.0, 1.0, roiTextName);
        }
        else {
            viewer->removeShape("roi_edge_y_box");
            viewer->removeShape("roi_edge_y_text");
        }
    }

    if (max_length_z > 0) {
        std::string line_id_z = "longest_line_z";
        viewer->removeShape(line_id_z);
        pcl::PointXYZ start_z(p1_z.x, p1_z.y, p1_z.z);
        pcl::PointXYZ end_z(p1_z.x, p1_z.y, p1_z.z + max_length_z);
        viewer->addLine(start_z, end_z, 0.0, 0.0, 1.0, line_id_z);

		viewer->setShapeRenderingProperties(
			pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
			3, line_id_z);

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

        struct CornerZ {
            float x, y, z;
        };

        CornerZ A = { p1_z.x, p2_z.y, p1_z.z };
        CornerZ B = { p1_z.x, p2_z.y, p1_z.z + max_length_z };

        float delta_z = 0.15f;

        float roi_min_x = A.x - delta_z;
        float roi_max_x = A.x + delta_z;
        float roi_min_y = A.y - delta_z;
        float roi_max_y = A.y + delta_z;
        float roi_min_z = std::min(A.z, B.z);
        float roi_max_z = std::max(A.z, B.z);

        int point_count = 0;
        float sumIntensity = 0.0f;

        if (enableINTENSITY) {
            for (const auto& pt : planeCloud->points) {
                if (pt.x >= roi_min_x && pt.x <= roi_max_x &&
                    pt.y >= roi_min_y && pt.y <= roi_max_y &&
                    pt.z >= roi_min_z && pt.z <= roi_max_z)
                {
                    point_count++;
                    sumIntensity += pt.intensity;
                }
            }
            float averageIntensity = (point_count > 0) ? (sumIntensity / point_count) : 0.0f;

            std::string roiCubeName = "roi_edge_z_box";
            viewer->removeShape(roiCubeName);
            viewer->addCube(roi_min_x, roi_max_x,
                roi_min_y, roi_max_y,
                roi_min_z, roi_max_z,
                1.0, 0.0, 1.0, roiCubeName);
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, roiCubeName);
            std::ostringstream textOss;
            textOss << "Count(Z): " << point_count << " | Intensity: " << averageIntensity;
            std::string roiTextName = "roi_edge_z_text";
            viewer->removeShape(roiTextName);
            viewer->addText3D(textOss.str(),
                pcl::PointXYZ((roi_min_x + roi_max_x) / 2,
                    (roi_min_y + roi_max_y) / 2,
                    (roi_min_z + roi_max_z) / 2),
                0.05, 1.0, 1.0, 1.0, roiTextName);
        }
        else {
            viewer->removeShape("roi_edge_z_box");
            viewer->removeShape("roi_edge_z_text");
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
            //std::cout << "Height" + std::to_string(x_index) + " : " + std::to_string(static_cast<int>(max_x_value * 1000)) << std::endl;

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

void heartbeatThreadFunction() {
    while (heartbeatRunning.load()) {
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
        std::this_thread::sleep_for(std::chrono::seconds(1)); // 1초마다 신호 전송
    }
}


// ----------------------------------------------------------------------------
static std::map<std::string, vtkImageData*> imageCache;
vtkImageData* LoadPNG(const std::string& path) {
    if (!imageCache.count(path)) {
        auto rdr = vtkSmartPointer<vtkPNGReader>::New();
        rdr->SetFileName(path.c_str());
        rdr->Update();
        imageCache[path] = rdr->GetOutput();
        imageCache[path]->Register(nullptr);
    }
    return imageCache[path];
}

vtkSmartPointer<vtkButtonWidget> MakeButton(
    vtkRenderWindowInteractor* iren,
    vtkRenderer* uiRen,
    const std::string& icon_off,
    const std::string& icon_on,
    double bnds[6],
    bool* flag_ptr,
    const std::string& flagName)     
{
    // 1) 버튼 리프레젠테이션
    auto rep = vtkSmartPointer<vtkTexturedButtonRepresentation2D>::New();
    rep->SetNumberOfStates(2);
    rep->SetButtonTexture(0, LoadPNG(icon_off));
    rep->SetButtonTexture(1, LoadPNG(icon_on));
    rep->SetRenderer(uiRen);
    rep->PlaceWidget(bnds);

    if (flag_ptr) {
        rep->SetState(*flag_ptr ? 1 : 0);
    }
    else {
        rep->SetState(0);
    }
    // 2) 버튼 위젯
    auto w = vtkSmartPointer<vtkButtonWidget>::New();
    w->SetInteractor(iren);
    w->SetCurrentRenderer(uiRen);
    w->SetRepresentation(rep);
    w->On();

    if (flag_ptr) {
        auto cb = vtkSmartPointer<ButtonCallback>::New();
        cb->toggleFlag = flag_ptr;
        cb->name = flagName;
        w->AddObserver(vtkCommand::StateChangedEvent, cb);
    }

    return w;
}

struct ReachCounterCallback : public vtkCommand {
    static ReachCounterCallback* New() { return new ReachCounterCallback; }
	WATAConfig* config = nullptr;
    bool*   toggleFlag = nullptr;
    float*  groundHeightPtr = nullptr;
    vtkTextActor* heightTextActor = nullptr;
    vtkButtonWidget* buttonWidget = nullptr;

    void Execute(vtkObject*, unsigned long, void*) override {
        config->flag_reach_off_counter = !config->flag_reach_off_counter;

        float h = config->flag_reach_off_counter ? (config->reach_height / 1000.0f) : (config->counterbalance_height / 1000.0f);
        *groundHeightPtr = h;

        auto rep = static_cast<vtkTexturedButtonRepresentation2D*>(
            buttonWidget->GetRepresentation());
        rep->SetState(config->flag_reach_off_counter ? 1 : 0);

        std::ostringstream ss;
        ss << "Current Ground Height: " << h << "m";
        heightTextActor->SetInput(ss.str().c_str());
    }
};



struct ExitCallback : public vtkCommand {
    static ExitCallback* New() { return new ExitCallback; }
    pcl::visualization::PCLVisualizer* viewer = nullptr;
    vtkRenderWindowInteractor* iren = nullptr;

    void Execute(vtkObject*, unsigned long, void*) override {
        // 1) VTK 인터랙터 정지
        if (iren) iren->TerminateApp();
        // 2) PCLVisualizer 루프 종료
        if (viewer) viewer->close();
        // 3) 전역 리소스 해제 (필요하면)
        LivoxLidarSdkUninit();
        curl_global_cleanup();
        std::exit(0);
    }
};

struct RebootCallback : public vtkCommand {
    static RebootCallback* New() { return new RebootCallback; }
    vtkRenderWindowInteractor* iren{ nullptr };
    pcl::visualization::PCLVisualizer* viewer{ nullptr };

    void Execute(vtkObject*, unsigned long, void*) override {

        if (iren)   iren->TerminateApp(); 
        if (viewer) viewer->close();
        rebootRequested.store(true);
    }
};

struct ResetCallback : public vtkCommand {
    static ResetCallback* New() { return new ResetCallback; }
    WATAConfig* cfg = nullptr;
    vtkSliderRepresentation2D* iterRep = nullptr;
    vtkSliderRepresentation2D* meanRep = nullptr;
    vtkSliderRepresentation2D* thrRep = nullptr;
    vtkSliderRepresentation2D* yaw0Rep = nullptr;
    vtkSliderRepresentation2D* yaw1Rep = nullptr;
    vtkSliderRepresentation2D* pit0Rep = nullptr;
    vtkSliderRepresentation2D* pit1Rep = nullptr;
    pcl::visualization::PCLVisualizer* viewer = nullptr;
    int     default_iter;
    int     default_mean_k;
    float   default_threshold;
    FovCfg  default_fov0;
    FovCfg  default_fov1;

    void Execute(vtkObject*, unsigned long, void*) override {
        // 1) WATAConfig 복원
        cfg->iteration = default_iter;
        cfg->mean_k = default_mean_k;
        cfg->threshold = default_threshold;
        // 2) 슬라이더 UI 복원
        iterRep->SetValue(default_iter);
        meanRep->SetValue(default_mean_k);
        thrRep->SetValue(default_threshold);
        // 3) FOV 슬라이더 복원
        yaw0Rep->SetValue(default_fov0.yaw_start);
        yaw1Rep->SetValue(default_fov0.yaw_stop);
        pit0Rep->SetValue(default_fov1.pitch_start);
        pit1Rep->SetValue(default_fov1.pitch_stop);
        // 4) SDK 에도 한번 더 보내기
        SetLivoxLidarFovCfg0(g_lidar_handle.load(), &default_fov0, nullptr, nullptr);
        SetLivoxLidarFovCfg1(g_lidar_handle.load(), &default_fov1, nullptr, nullptr);
        EnableLivoxLidarFov(g_lidar_handle.load(), 1, nullptr, nullptr);
        // 5) 카메라 리셋
        viewer->resetCamera();
    }
};

struct FovSliderCallback : public vtkCommand {
    static FovSliderCallback* New() { return new FovSliderCallback; }
    uint32_t handle;
    int field;         // 0=start_yaw, 1=stop_yaw, 2=start_pitch, 3=stop_pitch

    void Execute(vtkObject* caller, unsigned long, void*) override {
        auto slider = static_cast<vtkSliderWidget*>(caller);
        double v = static_cast<vtkSliderRepresentation*>(slider->GetRepresentation())->GetValue();

        // 1) FovCfg 에 값 채우기
        switch (field) {
        case 0: fov_cfg0.yaw_start = static_cast<int>(v); break;
        case 1: fov_cfg0.yaw_stop = static_cast<int>(v); break;
        case 2: fov_cfg1.pitch_start = static_cast<int>(v); break;
        case 3: fov_cfg1.pitch_stop = static_cast<int>(v); break;
        }
		uint32_t handle = g_lidar_handle.load();
		if (handle == 0) return;

        // 2) SDK 에 연속 전송
        SetLivoxLidarFovCfg0(handle, &fov_cfg0, nullptr, nullptr);
        SetLivoxLidarFovCfg1(handle, &fov_cfg1, nullptr, nullptr);

        // 3) 활성화
        EnableLivoxLidarFov(handle, /*fov_en=*/1, nullptr, nullptr);

        // (디버그)
        std::cout
            << "[FOV] yaw[" << fov_cfg0.yaw_start << "," << fov_cfg0.yaw_stop << "] "
            << "pit[" << fov_cfg1.pitch_start << "," << fov_cfg1.pitch_stop << "]\n";
    }
};

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, const char* argv[]) {
    vtkOutputWindow::GetInstance()->PromptUserOff();
    vtkObject::GlobalWarningDisplayOff();


    std::cout << "Current working directory: "
        << std::filesystem::current_path() << std::endl;

    curl_global_init(CURL_GLOBAL_ALL); // curl 전역 초기화 (프로그램 시작 시 한번 호출)

    lastLidarTimeMillis.store(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());




    // 1) 설정 로드
    const std::string LIVOX_PATH = "config/config.json";
    const std::string WATA_PATH = "config/dev_setting.json";
    WATAConfig config = readConfigFromJson(WATA_PATH);

    enableDetectPlaneYZ = config.flag_detect_plane_yz;
    enableLoadROI = config.flag_load_roi;
    enableRAWcloud = config.flag_raw_cloud;
    enableINTENSITY = config.flag_intensity;
    enableDualEmit = config.flag_dual_emit;
    onReachoffCounter = config.flag_reach_off_counter;
	fixed_ground_height = onReachoffCounter ? config.reach_height/1000.0f : config.counterbalance_height/1000.0f;
    enableReplay = config.flag_replay;
    enableHeartBeat = config.flag_heart_beat;

    const int   default_iteration = config.iteration;
    const int   default_mean_k = config.mean_k;
    const float default_threshold = config.threshold;
    const FovCfg default_fov0 = fov_cfg0;   // = {0,360,0,0}
    const FovCfg default_fov1 = fov_cfg1;   // = {0,0,-2,52}


    bool READ_PCD_FROM_FILE = config.read_file;
    bool SAVE_PCD_FROM_FILE = config.save_file;
    const std::string READ_PCD_FILE_NAME = config.read_file_name;
    const std::string SAVE_PCD_FILE_NAME = config.save_file_name;


    const int iteration = config.iteration;

    if (enableHeartBeat) {
        std::thread heartbeatThread(heartbeatThreadFunction);
        heartbeatThread.detach(); // 스레드를 분리하여 독립적으로 실행
    }

    const float ANGLE_DEGREES = 0.0;
    const float THETA = ANGLE_DEGREES * M_PI / 180.f;
    const float COS_THETA = std::cos(THETA);
    const float SIN_THETA = std::sin(THETA);

    if (READ_PCD_FROM_FILE) {
        // 파일에서 PCD 로드
        std::cout << "Reading point cloud: " << READ_PCD_FILE_NAME << std::endl;
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(READ_PCD_FILE_NAME, *cloud_loaded) == -1) {
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
    auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>("Volume Measurement Program(v1.11)");
    /*
    [Version 정리]   
    - version 1.1 : 초기 모델
	- version 1.11 : 초기 모델
        + 리치형vs카운터형 환경세팅 추가
			1. dev_setting.json에 "reachoffcounter": true/false 에 따라 지면 높이, 측정 범위 변경
    - 
    
    
    */
    
    viewer->addCoordinateSystem(1.0);
    viewer->setBackgroundColor(0.1, 0.1, 0.1);
    viewer->setCameraPosition(4.14367, 5.29453, -3.91817, 0.946026, -0.261667, 0.191218);

    int previous_pallet_index = 0;

    viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)viewer.get());


    x_lengths.clear();
    y_lengths.clear();
    z_lengths.clear();

    // V_start_process 상태 텍스트 초기화
    std::string initial_status = "V_start_process: " + std::string(V_start_process ? "True" : "False");
    viewer->addText(initial_status, 20, 630, 20, 1, 1, 1, "v_start_process_text");



    // 전역 변수 또는 메인 루프 상단에 추가
    bool previous_V_start_process = V_start_process;

    /*
    버튼 (on/off)
    - 적재물 roi ㅇ
    - cloud_pcd (raw 포인트) ㅇ
    - detectPlaneYZ ㅇ
    - 가능하면 intensity ㅇ

    슬라이더 조절 (파라미터 값, roi 범위 등)
    - iteration ㅇ

    - mean_k ㅇ
    - threshold ㅇ

    - height_threshold

	- 라이다 FPS (기본값:10Hz->100,1000Hz까지 조절가능하게)


    - roi
    - roi_y_start
    - roi_y_end
    - roi_z_start
    - roi_z_end
    - roi_angle
    - V_x_start
    - v_x_end



    */






    // 1) RenderWindow 세팅
    vtkRenderWindow* rw = viewer->getRenderWindow();

	rw->SetSize(1280, 720);      // 윈도우 크기
    rw->SetAlphaBitPlanes(1);   // 알파 채널 허용
    rw->SetMultiSamples(0);     // 멀티샘플링 해제
    rw->SetNumberOfLayers(3);

    // 2) 좌측(0.0~0.66) 메인 3D 렌더러
    auto mainRen = viewer->getRendererCollection()->GetFirstRenderer();
    mainRen->SetViewport(0.0, 0.0, 0.75, 1.0);
    mainRen->SetLayer(0);

    // 3) 우측(0.66~1.0) UI 전용 렌더러
    vtkSmartPointer<vtkRenderer> uiRen = vtkSmartPointer<vtkRenderer>::New();
    uiRen->SetViewport(0.75, 0.0, 1.0, 1.0);
    uiRen->InteractiveOff();
    uiRen->SetLayer(1);
    rw->AddRenderer(uiRen);


    vtkSmartPointer<vtkTextActor> devPanel = vtkSmartPointer<vtkTextActor>::New();
	devPanel->GetTextProperty()->SetFontSize(18);
    devPanel->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
    devPanel->GetTextProperty()->SetBackgroundColor(0.0, 0.0, 0.0);
    devPanel->GetTextProperty()->SetBackgroundOpacity(0.6);
    
    devPanel->SetInput("");
	devPanel->SetPosition(770, 10);
    mainRen->AddActor2D(devPanel);

    vtkSmartPointer<vtkTextActor> StatusPanel = vtkSmartPointer<vtkTextActor>::New();
	StatusPanel->GetTextProperty()->SetFontSize(16);
	StatusPanel->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
	StatusPanel->GetTextProperty()->SetBackgroundColor(0.0, 0.0, 0.0);
	StatusPanel->GetTextProperty()->SetBackgroundOpacity(0.6);
    mainRen->AddActor2D(StatusPanel);



    vtkNew<vtkRenderer> logoRen;
    logoRen->SetLayer(2);
    logoRen->InteractiveOff();
    logoRen->SetViewport(0.00, 0.00, 0.10, 0.10);
    logoRen->SetBackgroundAlpha(0.0);
    rw->AddRenderer(logoRen);

    // 3) PNG 리더로 로고 이미지 로드
    vtkNew<vtkPNGReader> logoReader;
    logoReader->SetFileName("icons/logo.png");
    logoReader->Update();

    vtkNew<vtkImageSliceMapper> sliceMapper;
    sliceMapper->SetInputConnection(logoReader->GetOutputPort());

    vtkNew<vtkImageSlice> slice;
    slice->SetMapper(sliceMapper);

    logoRen->AddViewProp(slice);





    uiRen->SetPreserveColorBuffer(false);
    uiRen->SetPreserveDepthBuffer(false);

    // 4) 인터랙터
    vtkRenderWindowInteractor* iren = rw->GetInteractor();




// ----------------------[버튼]----------------------

    // 윈도우 크기 받아오기
    int* winSize = rw->GetSize();
    constexpr int ICON_PX = 128;

    // 버튼 정보만 정의
    struct BtnInfo {
        std::string off, on;
        double x_norm, y_norm;
        bool* flag;
        std::string name;
    };

    vtkSmartPointer<vtkTextActor> currentGroundText = vtkSmartPointer<vtkTextActor>::New();
    currentGroundText->GetTextProperty()->SetFontSize(16);
    currentGroundText->SetPosition(20, 660);
    uiRen->AddActor2D(currentGroundText);

    currentGroundText->SetInput(
        ("Current Ground Height: " + std::to_string(fixed_ground_height) + "m")
        .c_str());


    std::vector<BtnInfo> btns = {
        { "icons/icon_yz_off_black_filled.png",   "icons/icon_yz_on_black_filled.png",   0.00, 0.95, &config.flag_detect_plane_yz , "detectPlaneYZ"},
        { "icons/icon_roi_off_black_filled.png",  "icons/icon_roi_on_black_filled.png",  0.05, 0.95, &config.flag_load_roi, "LoadROI"},
        { "icons/icon_raw_off_black_filled (2).png",  "icons/icon_raw_on_black_filled.png",  0.10, 0.95, &config.flag_raw_cloud , "RAWcloud"},
        { "icons/icon_intensity_off_reflection.png","icons/icon_intensity_on_reflection.png",0.15,0.95, &enableINTENSITY     , "intesity"},
        { "icons/icon_dual_emit_off_black_filled.png","icons/icon_dual_emit_on_black_filled.png",0.00,0.861,&config.flag_dual_emit , "DualEmit"},
        { "icons/icon_forklift_counter.png",       "icons/icon_forklift_reach.png",       0.15, 0.861, &onReachoffCounter, "Reach <-> CounterBalace"},
        { "icons/icon_play.png",                   "icons/icon_stop.png",                 0.05, 0.861, &is_paused , "Stop or Play"},
        { "icons/icon_loop_off_black_filled.png",  "icons/icon_loop_on_black_filled.png", 0.10, 0.861, &config.flag_replay , "Replay"},
        { "icons/icon_heartbeat_off_black_filled.png",  "icons/icon_heartbeat_on_black_filled.png", 0.00, 0.772,& config.flag_heart_beat , "HeartBeat" }
    };
    int reachBtnIndex = 0;
    for (int i = 0; i < btns.size(); ++i) {
        if (btns[i].name == "Reach <-> CounterBalace") {
            reachBtnIndex = i;
            break;
        }
    }
    std::vector<vtkSmartPointer<vtkButtonWidget>> buttonWidgets;


    // 루프 한 번에 모두 생성
    for (auto& bi : btns) {
        double x0 = winSize[0] * bi.x_norm;
        double x1 = x0 + ICON_PX;
        double y1 = winSize[1] * bi.y_norm;
        double y0 = y1 - ICON_PX;
        double bnds[6] = { x0, x1, y0, y1, 0.0, 0.0 };
        auto btn = MakeButton(iren, uiRen, bi.off, bi.on, bnds, bi.flag, bi.name);
        buttonWidgets.push_back(btn);
    }

    auto reachCb = vtkSmartPointer<ReachCounterCallback>::New();
    reachCb->config = &config;
    reachCb->toggleFlag = &onReachoffCounter;
    reachCb->groundHeightPtr = &fixed_ground_height;
    reachCb->heightTextActor = currentGroundText.Get();
    reachCb->buttonWidget = buttonWidgets[reachBtnIndex].Get();
    buttonWidgets[reachBtnIndex]
        ->AddObserver(vtkCommand::StateChangedEvent, reachCb);


    auto exitRep = vtkSmartPointer<vtkTexturedButtonRepresentation2D>::New();
    exitRep->SetNumberOfStates(1);
    exitRep->SetButtonTexture(0, LoadPNG("icons/exit.png"));  // exit.png 하나만
	double exit_x0 = winSize[0] * 0.20;
	double exit_x1 = exit_x0 + 128;
	double exit_y1 = winSize[1] * 1.05;
	double exit_y0 = exit_y1 - 64;
	double exit_bnds[6] = { exit_x0, exit_x1, exit_y0, exit_y1, 0.0, 0.0 };
    exitRep->PlaceWidget(exit_bnds);  // bnds 는 원하는 위치/크기로 설정한 double[6]
    exitRep->SetRenderer(uiRen);

    // 버튼 위젯 생성
    auto exitBtn = vtkSmartPointer<vtkButtonWidget>::New();
    exitBtn->SetInteractor(iren);
    exitBtn->SetCurrentRenderer(uiRen);
    exitBtn->SetRepresentation(exitRep);
    exitBtn->On();

    // 콜백 연결
    auto cbExit = vtkSmartPointer<ExitCallback>::New();
    cbExit->iren = iren;
    cbExit->viewer = viewer.get();
    exitBtn->AddObserver(vtkCommand::StateChangedEvent, cbExit);


    // 2) Reboot 버튼 표현 생성
    auto rebootRep = vtkSmartPointer<vtkTexturedButtonRepresentation2D>::New();
    rebootRep->SetNumberOfStates(1);
    rebootRep->SetButtonTexture(0, LoadPNG("icons/reboot.png"));  // 아이콘 파일
    // 화면 우측 하단 어딘가 적당히 배치
    double bx0 = winSize[0] * 0.124;
    double bx1 = bx0 + 128;
    double by1 = winSize[1] * 1.05;
    double by0 = by1 - 64;
    double bnds[6] = { bx0, bx1, by0, by1, 0, 0 };
    rebootRep->PlaceWidget(bnds);
    rebootRep->SetRenderer(uiRen);

    // 3) Reboot 버튼 위젯 생성·등록
    auto rebootBtn = vtkSmartPointer<vtkButtonWidget>::New();
    rebootBtn->SetInteractor(iren);
    rebootBtn->SetCurrentRenderer(uiRen);
    rebootBtn->SetRepresentation(rebootRep);
    rebootBtn->On();

    // 4) 콜백 달기
    auto cbReboot = vtkSmartPointer<RebootCallback>::New();
    cbReboot->iren = iren;
    cbReboot->viewer = viewer.get();

    rebootBtn->AddObserver(vtkCommand::StateChangedEvent, cbReboot);





 //------------------------[슬라이더]--------------------------

    //-------------------------[iteration]--------------------------
    vtkSmartPointer<vtkSliderRepresentation2D> iteration_sliderRep =
        vtkSmartPointer<vtkSliderRepresentation2D>::New();
    iteration_sliderRep->SetMinimumValue(250);
    iteration_sliderRep->SetMaximumValue(1000);
    iteration_sliderRep->SetValue(config.iteration);
    iteration_sliderRep->SetTitleText("iteration");
	iteration_sliderRep->SetEndCapLength(0.01);
	iteration_sliderRep->SetEndCapWidth(0.01);
    iteration_sliderRep->GetPoint1Coordinate()->SetCoordinateSystemToNormalizedDisplay();
    iteration_sliderRep->GetPoint1Coordinate()->SetValue(0.05, 0.55);
    iteration_sliderRep->GetPoint2Coordinate()->SetCoordinateSystemToNormalizedDisplay();
    iteration_sliderRep->GetPoint2Coordinate()->SetValue(0.20, 0.55);
    iteration_sliderRep->GetSliderProperty()->SetColor(1.0, 0.0, 0.0);
    iteration_sliderRep->GetSelectedProperty()->SetColor(1.0, 1.0, 0.0);
    iteration_sliderRep->SetTitleHeight(0.02);
    iteration_sliderRep->SetLabelHeight(0.015);
    iteration_sliderRep->SetSliderLength(0.03);
    iteration_sliderRep->SetSliderWidth(0.015);
    iteration_sliderRep->SetTubeWidth(0.005);
    iteration_sliderRep->SetRenderer(uiRen);

    vtkSmartPointer<vtkSliderWidget> iteration_sliderWidget =
        vtkSmartPointer<vtkSliderWidget>::New();
    iteration_sliderWidget->SetInteractor(iren);
    iteration_sliderWidget->SetCurrentRenderer(uiRen);
    iteration_sliderWidget->SetRepresentation(iteration_sliderRep);
    iteration_sliderWidget->SetAnimationModeToAnimate();
    iteration_sliderWidget->KeyPressActivationOff();
    iteration_sliderWidget->SetEnabled(1);

    //iteration_sliderWidget->On();
    auto iteration_scb = vtkSmartPointer<SliderCallback>::New();
    iteration_scb->iteration_ptr = &config;
    iteration_sliderWidget->AddObserver(vtkCommand::InteractionEvent, iteration_scb);


	//-------------------------[mean_k]--------------------------
    vtkSmartPointer<vtkSliderRepresentation2D> mean_k_sliderRep =
        vtkSmartPointer<vtkSliderRepresentation2D>::New();
    mean_k_sliderRep->SetMinimumValue(1);
    mean_k_sliderRep->SetMaximumValue(100);
    mean_k_sliderRep->SetValue(config.mean_k);
    mean_k_sliderRep->SetTitleText("mean_k");
	mean_k_sliderRep->SetEndCapLength(0.01);
	mean_k_sliderRep->SetEndCapWidth(0.01);
    mean_k_sliderRep->GetPoint1Coordinate()->SetCoordinateSystemToNormalizedDisplay();
    mean_k_sliderRep->GetPoint1Coordinate()->SetValue(0.05, 0.45);
    mean_k_sliderRep->GetPoint2Coordinate()->SetCoordinateSystemToNormalizedDisplay();
    mean_k_sliderRep->GetPoint2Coordinate()->SetValue(0.20, 0.45);
	mean_k_sliderRep->GetSliderProperty()->SetColor(1.0, 0.0, 0.0);
	mean_k_sliderRep->GetSelectedProperty()->SetColor(1.0, 1.0, 0.0);
	mean_k_sliderRep->SetTitleHeight(0.02);
	mean_k_sliderRep->SetLabelHeight(0.015);
	mean_k_sliderRep->SetSliderLength(0.03);
	mean_k_sliderRep->SetSliderWidth(0.015);
    mean_k_sliderRep->SetTubeWidth(0.005);
    mean_k_sliderRep->SetRenderer(uiRen);

    vtkSmartPointer<vtkSliderWidget> mean_k_sliderWidget =
        vtkSmartPointer<vtkSliderWidget>::New();
    mean_k_sliderWidget->SetInteractor(iren);
    mean_k_sliderWidget->SetCurrentRenderer(uiRen);
    mean_k_sliderWidget->SetRepresentation(mean_k_sliderRep);
    mean_k_sliderWidget->SetAnimationModeToAnimate();
    mean_k_sliderWidget->KeyPressActivationOff();
    mean_k_sliderWidget->SetEnabled(1);

    //sliderWidget->On();
    vtkSmartPointer<SliderCallback> scb = vtkSmartPointer<SliderCallback>::New();
    scb->meank_ptr = &config;
    mean_k_sliderWidget->AddObserver(vtkCommand::InteractionEvent, scb);

	// -------------------------[threshold]--------------------------
	vtkSmartPointer<vtkSliderRepresentation2D> threshold_sliderRep =
		vtkSmartPointer<vtkSliderRepresentation2D>::New();
	threshold_sliderRep->SetMinimumValue(0.5);
	threshold_sliderRep->SetMaximumValue(2.5);
	threshold_sliderRep->SetValue(config.threshold);
	threshold_sliderRep->SetTitleText("threshold");
	threshold_sliderRep->SetEndCapLength(0.01);
	threshold_sliderRep->SetEndCapWidth(0.01);
	threshold_sliderRep->GetPoint1Coordinate()->SetCoordinateSystemToNormalizedDisplay();
	threshold_sliderRep->GetPoint1Coordinate()->SetValue(0.05, 0.35);
	threshold_sliderRep->GetPoint2Coordinate()->SetCoordinateSystemToNormalizedDisplay();
	threshold_sliderRep->GetPoint2Coordinate()->SetValue(0.20, 0.35);
	threshold_sliderRep->GetSliderProperty()->SetColor(1.0, 0.0, 0.0);
	threshold_sliderRep->GetSelectedProperty()->SetColor(1.0, 1.0, 0.0);
	threshold_sliderRep->SetTitleHeight(0.02);
	threshold_sliderRep->SetLabelHeight(0.015);
	threshold_sliderRep->SetSliderLength(0.03);
	threshold_sliderRep->SetSliderWidth(0.015);
	threshold_sliderRep->SetTubeWidth(0.005);
	threshold_sliderRep->SetRenderer(uiRen);
	vtkSmartPointer<vtkSliderWidget> threshold_sliderWidget =
		vtkSmartPointer<vtkSliderWidget>::New();
	threshold_sliderWidget->SetInteractor(iren);
	threshold_sliderWidget->SetCurrentRenderer(uiRen);
	threshold_sliderWidget->SetRepresentation(threshold_sliderRep);
	threshold_sliderWidget->SetAnimationModeToAnimate();
	threshold_sliderWidget->KeyPressActivationOff();
	threshold_sliderWidget->SetEnabled(1);

	//threshold_sliderWidget->On();
	vtkSmartPointer<SliderCallback> threshold_scb = vtkSmartPointer<SliderCallback>::New();
	threshold_scb->threshold_ptr = &config;
	threshold_sliderWidget->AddObserver(vtkCommand::InteractionEvent, threshold_scb);


    std::vector<vtkSmartPointer<vtkSliderWidget>> fovSliderWidgets;
    vtkSmartPointer<vtkSliderRepresentation2D> yawStartRep;
    vtkSmartPointer<vtkSliderRepresentation2D> yawStopRep;
    vtkSmartPointer<vtkSliderRepresentation2D> pitStartRep;
    vtkSmartPointer<vtkSliderRepresentation2D> pitStopRep;

    // 수평 시작
    auto makeFovSlider = [&](double minV, double maxV, double yNorm,
        const char* title, int field,
        int initialValue,
        vtkSmartPointer<vtkSliderRepresentation2D>& outRep) {
        auto rep = vtkSmartPointer<vtkSliderRepresentation2D>::New();
        rep->SetMinimumValue(minV);
        rep->SetMaximumValue(maxV);
        rep->SetValue(initialValue); 
        rep->SetTitleText(title);
        rep->SetEndCapLength(0.01);
        rep->SetEndCapWidth(0.01);
        rep->GetSliderProperty()->SetColor(1.0, 0.0, 0.0);
        rep->GetSelectedProperty()->SetColor(1.0, 1.0, 0.0);
        rep->SetTitleHeight(0.02);
        rep->SetLabelHeight(0.015);
        rep->SetSliderLength(0.03);
        rep->SetSliderWidth(0.015);
        rep->SetTubeWidth(0.005);

        rep->GetPoint1Coordinate()->SetCoordinateSystemToNormalizedDisplay();
        rep->GetPoint2Coordinate()->SetCoordinateSystemToNormalizedDisplay();
        rep->GetPoint1Coordinate()->SetValue(0.05, yNorm);
        rep->GetPoint2Coordinate()->SetValue(0.20, yNorm);
        rep->SetRenderer(uiRen);

        auto widget = vtkSmartPointer<vtkSliderWidget>::New();
        widget->SetInteractor(iren);
        widget->SetCurrentRenderer(uiRen);
        widget->SetRepresentation(rep);
        widget->SetAnimationModeToAnimate();
        widget->KeyPressActivationOff();
        widget->SetEnabled(1);
        widget->On();



        auto cb = vtkSmartPointer<FovSliderCallback>::New();
        cb->handle = g_lidar_handle.load();
        cb->field = field;
        widget->AddObserver(vtkCommand::InteractionEvent, cb);

        fovSliderWidgets.push_back(widget);

        outRep = rep;
    };

    // 네 개의 슬라이더 달기
    makeFovSlider(0, 360, 0.25, "Yaw Start", 0, 0, yawStartRep);
    makeFovSlider(0, 360, 0.20, "Yaw Stop", 1, 360, yawStopRep);
    makeFovSlider(-2, 52, 0.15, "Pitch Start", 2, -2, pitStartRep);
    makeFovSlider(-2, 52, 0.10, "Pitch Stop", 3, 52, pitStopRep);


    // ── 리셋 버튼 표현 ──
    auto resetRep = vtkSmartPointer<vtkTexturedButtonRepresentation2D>::New();
    resetRep->SetNumberOfStates(1);
    resetRep->SetButtonTexture(0, LoadPNG("icons/reset.png"));
    double rx0 = winSize[0] * 0.075, rx1 = rx0 + 128;
    double ry1 = winSize[1] * 1.05, ry0 = ry1 - 64;
    double rb[6] = { rx0,rx1, ry0,ry1, 0,0 };
    resetRep->PlaceWidget(rb);
    resetRep->SetRenderer(uiRen);

    auto resetBtn = vtkSmartPointer<vtkButtonWidget>::New();
    resetBtn->SetInteractor(iren);
    resetBtn->SetCurrentRenderer(uiRen);
    resetBtn->SetRepresentation(resetRep);
    resetBtn->On();

    // 콜백 연결
    auto cbReset = vtkSmartPointer<ResetCallback>::New();
    cbReset->cfg = &config;
    cbReset->iterRep = iteration_sliderRep;
    cbReset->meanRep = mean_k_sliderRep;
    cbReset->thrRep = threshold_sliderRep;
    cbReset->yaw0Rep = vtkSliderRepresentation2D::SafeDownCast(
        fovSliderWidgets[0]->GetRepresentation());
    cbReset->yaw1Rep = vtkSliderRepresentation2D::SafeDownCast(
        fovSliderWidgets[1]->GetRepresentation());
    cbReset->pit0Rep = vtkSliderRepresentation2D::SafeDownCast(
        fovSliderWidgets[2]->GetRepresentation());
    cbReset->pit1Rep = vtkSliderRepresentation2D::SafeDownCast(
        fovSliderWidgets[3]->GetRepresentation());
    cbReset->viewer = viewer.get();
    cbReset->default_iter = default_iteration;
    cbReset->default_mean_k = default_mean_k;
    cbReset->default_threshold = default_threshold;
    cbReset->default_fov0 = default_fov0;
    cbReset->default_fov1 = default_fov1;
    resetBtn->AddObserver(vtkCommand::StateChangedEvent, cbReset);

    viewer->getRenderWindow()->Render();

    bool prev_onReach = onReachoffCounter;

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



            std::ostringstream ss;
            if (READ_PCD_FROM_FILE) { // 파일에서 PCD 로드
                std::lock_guard<std::mutex> lk(control_mutex);
                size_t total = cloud_loaded->points.size();
                float pct = total > 0 ? (100.0f * point_index) / total : 0.0f;
                ss << "Mode: Read\n(" << READ_PCD_FILE_NAME << ") "
                    << std::fixed << std::setprecision(1) << pct << "%"
                    << "\n" << (enableReplay ? " [Loop:On]" : "[Loop:Off]");

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
                if (point_index >= static_cast<int>(total)) {
                    if (enableReplay) {
                        point_index = 0;
                    }
                    else { reading_active = false; }
                }
            }
            else if (SAVE_PCD_FROM_FILE) {
                ss << "Mode: Save\n(" << SAVE_PCD_FILE_NAME << ") ";
            }
            else { ss << "Mode: Real-time"; }

            if (heightCalibration_mode) { // 지면 캘리브레이션 모드                
                {
                    std::lock_guard<std::mutex> lock(g_mutex);
                    cloud_ground->clear();

                    pcl::PointCloud<pcl::PointXYZI>::Ptr src_cloud;
                    if (READ_PCD_FROM_FILE) {
                        src_cloud = cloud_merge;
                    }
                    else {
                        src_cloud = cloud_merge;
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
                    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>
                        color_handler(cloud_ground, 255, 0, 0);
                    viewer->addPointCloud<pcl::PointXYZI>(cloud_ground, color_handler, "roi_cloud");
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
			else if (!heightCalibration_mode && ground_height_fixed) {
				// 지면 높이 고정 상태에서 지면 캘리브레이션 모드 종료
				heightCalibration_mode = false;
				ground_height_fixed = false;
				std::cout << "[Calibration] Height Calibration mode ended." << std::endl;
				viewer->removeShape("ground_result_text");
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

                devPanel->SetInput("");

                // 포인트 클라우드 데이터 초기화
                resetPointCloudData();

                // 이전 상태 업데이트
                previous_V_start_process = V_start_process;

                // Viewer에 현재 상태 텍스트 업데이트
                std::string status_text = "V_start_process: " + std::string(V_start_process ? "True" : "False");
                viewer->removeShape("v_start_process_text");
                viewer->addText(status_text, 20, 630, 20, 1, 1, 1, "v_start_process_text");

                std::cout << "[INFO] V_start_process state changed to "
                    << (V_start_process ? "True" : "False") << ". Data has been reset." << std::endl;
            }



            if (cloud_merge && !cloud_merge->empty()) {
                std::lock_guard<std::mutex> lk(g_mutex);

                if (config.flag_raw_cloud)
                {
                    viewer->removePointCloud("raw");
                    // 최신 cloud_raw 를 흰색 점으로 그리기
                    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rawHandler(cloud_merge, 255, 255, 255);
                    viewer->addPointCloud<pcl::PointXYZI>(cloud_merge, rawHandler, "raw");
                    viewer->setPointCloudRenderingProperties(
                        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "raw");
                }
                if (!config.flag_raw_cloud)
                {
                    viewer->removePointCloud("raw");
                }



                // ------------------------------------------------------------------
                // 1층 픽업


                int count_load_roi_1 = 0;


                float loadROI_x1_min = -fixed_ground_height + 0.1f;
                float loadROI_x1_max = loadROI_x1_min + 1.9f;
                float loadROI_y_min = -0.9f;
				float loadROI_y_max = config.flag_reach_off_counter ? -0.45f : -0.20f;
                float loadROI_z_min = config.flag_reach_off_counter ? 0.35f : 0.20f;
                float loadROI_z_max = config.flag_reach_off_counter ? 0.65 : 0.50f;

                pcl::PointCloud<pcl::PointXYZI>::Ptr Pickup_cloud;

                if (READ_PCD_FROM_FILE) {
                    Pickup_cloud = cloud_merge;
                }
                else {
                    Pickup_cloud = cloud_merge;
                }

                for (const auto& point : Pickup_cloud->points) {
                    if (point.x >= loadROI_x1_min && point.x <= loadROI_x1_max &&
                        point.y >= loadROI_y_min && point.y <= loadROI_y_max &&
                        point.z >= loadROI_z_min && point.z <= loadROI_z_max) {
                        count_load_roi_1++;
                    }
                }


                if (count_load_roi_1 >= 10 && count_load_roi_1 <= 1000) {
                    PickUp_1 = true;
                    //std::cout << "[PICKUP] 1st Floor !!!" << std::endl;

                    std::ostringstream oss;
					oss << "Pickup 1st Floor: " << count_load_roi_1 << " points";
                    viewer->removeShape("pickup_text");
                    viewer->addText(oss.str(), 20, 350, 20, 0.0, 1.0, 0.0, "pickup_text");
                    viewer->setShapeRenderingProperties(
                        pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                        8, "pickup_text");



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
                    viewer->removeShape("pickup_text");

                    //viewer->removeShape("load_roi_box_1");
                }


                float roiVolXMin = -fixed_ground_height + 0.25f;
                float roiVolXMax = roiVolXMin + 1.9f;
                float roiVolYMin = config.flag_reach_off_counter ? -1.65f : -1.40f;
                float roiVolYMax = config.flag_reach_off_counter ? -0.45f : -0.20f;
                float roiVolZMin = config.flag_reach_off_counter ? -0.20f : -0.25f;
                float roiVolZMax = config.flag_reach_off_counter ? 1.20f : 1.25f;


                // 4) LoadROI 박스 그리기
                if (config.flag_load_roi) {
                    viewer->removeShape("load_roi_box");
                    viewer->addCube(
                        roiVolXMin, roiVolXMax,
                        roiVolYMin, roiVolYMax,
                        roiVolZMin, roiVolZMax,
                        0.0, 0.5, 1.0,
                        "load_roi_box");
                    viewer->setShapeRenderingProperties(
                        pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                        pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
                        "load_roi_box");
                    viewer->setShapeRenderingProperties(
                        pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                        3, "load_roi_box");
                }
                else {
                    viewer->removeShape("load_roi_box");
                }


                // -------------------------------------------------------------------------------------
                // [모드1] 부피 형상 측정
                // -------------------------------------------------------------------------------------
                if (V_start_process && PickUp_1) {
                    // V_start_process가 true인 경우: x < 0.0f인 포인트 처리
                    float start_min_y = std::numeric_limits<float>::infinity();
                    pcl::PointXYZI start_min_y_point;
                    float end_min_y = std::numeric_limits<float>::infinity();
                    pcl::PointXYZI end_min_y_point;
                    std::vector<float> x_values;

                    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_pcd_local(new pcl::PointCloud<pcl::PointXYZI>);
                    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_volume(new pcl::PointCloud<pcl::PointXYZI>);
                    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_angle_filtered(new pcl::PointCloud<pcl::PointXYZI>);

                    // 2) 클라우드와 벡터 초기화 및 reserve
                    cloud_filtered_volume->points.clear();
                    cloud_pcd_local->points.clear();
                    x_values.clear();

                    size_t N = cloud_merge->points.size();
                    cloud_filtered_volume->points.reserve(N);
                    cloud_pcd_local->points.reserve(N);
                    x_values.reserve(N);

                    // 3) 포인트 클라우드 필터링
                    for (auto& temp : cloud_merge->points) {
                        pcl::PointXYZI point;
                        point.x = temp.x;
                        point.y = temp.y * COS_THETA - temp.z * SIN_THETA;
                        point.z = temp.y * SIN_THETA + temp.z * COS_THETA;
                        point.intensity = temp.intensity; // intensity 필드도 복사


                        if (point.x >= roiVolXMin && point.x < roiVolXMax &&
                            point.y >= roiVolYMin && point.y <= roiVolYMax &&
                            point.z >= roiVolZMin && point.z <= roiVolZMax)
                        {
                            x_values.push_back(point.x);
                            cloud_filtered_volume->points.push_back(point);
                            cloud_pcd_local->points.push_back(point);
                        }
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
                


                    cloud_filtered_volume->width = cloud_filtered_volume->points.size();
                    cloud_filtered_volume->height = 1;

                    if (!config.flag_detect_plane_yz) {
                        viewer->removePointCloud("cloud_filtered_volume");
                        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> color_handler(
                            cloud_filtered_volume, 255, 255, 255);
                        viewer->addPointCloud<pcl::PointXYZI>(cloud_filtered_volume, color_handler, "cloud_filtered_volume");
                        viewer->setPointCloudRenderingProperties(
                            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_filtered_volume"
                        );
                    }
					//viewer->removePointCloud("cloud_pcd");
					//viewer->addPointCloud<pcl::PointXYZI>(cloud_pcd_local, "cloud_pcd");

                    if (!x_values.empty()) {
                        calcMaxX(x_values, max_x_value);
                        for (auto& temp : cloud_filtered_volume->points) {
                            temp.x = max_x_value;
                        }
                    }


                    // Voxel Downsample
                    voxelizePointCloud(cloud_filtered_volume, 0.05f, 0.02f, 0.02f);

                    // Outlier Remove
                    removeOutliers(cloud_filtered_volume, config);

                    // Angle Points 계산
                    //calculateAnglePoints(start_min_y_point, end_min_y_point, viewer);

                    // 추가 각도 보정 (재계산 필요)
                    float COS_THETA_updated = cos(angle_degrees * M_PI / 180.0);
                    float SIN_THETA_updated = sin(angle_degrees * M_PI / 180.0);
                    for (auto& temp : cloud_filtered_volume->points) {
                        pcl::PointXYZI point;
                        point.x = temp.x;
                        point.y = temp.y * COS_THETA_updated - temp.z * SIN_THETA_updated;
                        point.z = temp.y * SIN_THETA_updated + temp.z * COS_THETA_updated;
                        point.intensity = temp.intensity; // intensity 값도 복사!

                        cloud_angle_filtered->points.push_back(point);
                    }
                    //cloud_angle_filtered->width = static_cast<uint32_t>(cloud_angle_filtered->points.size());
                    //cloud_angle_filtered->height = 1;
                    //cloud_angle_filtered->is_dense = false;




                    //detectPlaneYZ(cloud_filtered_volume, viewer);
                    if (config.flag_detect_plane_yz) {
                        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> ch(
                            cloud_angle_filtered, 172, 255, 142);




                        viewer->removePointCloud("cloud_angle_filtered");
                        viewer->addPointCloud(cloud_angle_filtered, ch, "cloud_angle_filtered");

                        viewer->setPointCloudRenderingProperties(
                            pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                            2, "cloud_angle_filtered");



                        std::cout << "[DEBUG] cloud_angle_filtered size: "
                            << cloud_angle_filtered->points.size() << std::endl;

                        detectPlaneYZ(cloud_filtered_volume, viewer);
                    }
					else {
						viewer->removePointCloud("cloud_angle_filtered");
					}


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
                        ground_correction_mm = fixed_ground_height * 1000.0f;
                        result_height += (ground_correction_mm + 9.5f);

                        //if (ground_height_fixed) {
                        //    ground_correction_mm = fixed_ground_height * 1000.0f;
                        //    result_height += (ground_correction_mm + 9.5f);
                        //}
                        //else {
                        //    result_height += config.height_threshold + 9.5f;  // 기존 임시 보정 (2755)
                        //}

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
                            << "Angle: " << angle_degrees << " deg \n";

                        std::string result = oss.str();

						devPanel->SetInput(result.c_str());

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
					ss << cloud_pcd->size() << " pts";
                }

                // 파일 모드라면 다음 chunk 대기 위해 처리 종료
                if (READ_PCD_FROM_FILE) {
                    cloud_merge->clear();
                }

                StatusPanel->SetInput(ss.str().c_str());
                StatusPanel->SetPosition(550, 650);

                rw->Render();

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
    curl_global_cleanup();

    if (rebootRequested.load()) {
#if defined(_WIN32)
        // 현재 커맨드라인 그대로 다시 띄우기
        STARTUPINFOA si = { sizeof(si) };
        PROCESS_INFORMATION pi;
        // GetCommandLineA() 로 똑같은 파라미터 라인 가져오기
        char cmdLine[1024];
        strcpy_s(cmdLine, GetCommandLineA());
        if (CreateProcessA(nullptr, cmdLine,
            nullptr, nullptr, FALSE,
            0, nullptr, nullptr,
            &si, &pi)) {
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
        }
#else
        // execv 는 argv[0] 과 끝에 NULL 포인터를 넘겨야 합니다.
        char* const exec_args[] = { const_cast<char*>(argv[0]), nullptr };
        execv(argv[0], exec_args);
#endif
    }

    return 0;
}
