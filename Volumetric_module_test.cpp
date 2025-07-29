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
#include <Ws2tcpip.h>
#else
#include <unistd.h>
#include <arpa/inet.h>
#include <netdb.h>
#endif

#include "json.hpp" 

#include <regex>
#include <sstream>
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <limits>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <fstream>
#include <algorithm>
#include <cmath>

#include <curl/curl.h>

#include <omp.h> // 병렬처리
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
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/common/distances.h>

#include <vtkOutputWindow.h>
#include <vtkObject.h>

#include <unordered_map>
#include <Eigen/Dense>



#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>   // imwrite
#include <opencv2/highgui.hpp>

#include "resource_dev.h"

#include "config.hpp"
#include "vtk_ui.hpp"







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





// 호스트(PC) IP 가져오는 유틸
std::string GetHostIP() {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == -1) return "";
    struct hostent* he = gethostbyname(hostname);
    if (!he || he->h_addr_list[0] == nullptr) return "";
    return inet_ntoa(*(struct in_addr*)he->h_addr_list[0]);
}




// ----------------------------------------------------------------------------
// 전역
// ----------------------------------------------------------------------------
std::mutex control_mutex;
std::mutex g_mutex;
std::atomic<long long> lastLidarTimeMillis(0);
std::atomic<bool> heartbeatRunning(true);
std::atomic<bool> vtk_ui::rebootRequested{ false };


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

std::vector<std::string> volume_line_ids;


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
  0.0f,     // yaw_start = 0°
  360.0f,   // yaw_stop  = 360°
  -9.0f,52.0f
};
//FovCfg fov_cfg0 = {    // yaw_start(안씀), yaw_stop(안씀), pitch_start, pitch_stop
//  0,0,
//  -2,    // pitch_start = -2°
//  52     // pitch_stop  = +52°
//};


std::atomic<uint32_t> g_lidar_handle{ 0 };


bool enableDetectPlaneYZ = false;
bool enableLoadROI = false;
bool enableRAWcloud = false;
bool enableINTENSITY = false;
bool enableHeightOnly = false;
bool onReachoffCounter = false;
bool enableReplay = false;
bool enableHeartBeat = true;
bool enableVolume = false;
bool enableTuning = false;




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








// ----------------------------------------------------------------------------
// 리소스 콜백 (실시간 모드)
// ----------------------------------------------------------------------------
void PointCloudCallback(uint32_t handle, const uint8_t dev_type, LivoxLidarEthernetPacket* data, void* client_data) {
    if (g_lidar_handle.load() == 0) {
        g_lidar_handle = handle;

        SetLivoxLidarFovCfg0(handle, &fov_cfg0, nullptr, nullptr);
        //SetLivoxLidarFovCfg1(handle, &fov_cfg1, nullptr, nullptr);
        EnableLivoxLidarFov(handle, 1, nullptr, nullptr);
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
            viewer->addText(initial_status, 20, 530, 20, 1, 1, 1, "v_start_process_text");
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
            ec.setMaxClusterSize(20000);
            seg.setMaxIterations(10000);  // 기본값 1000보다 증가
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



// 멤버 변수
std::vector<float> width_samples, depth_samples, height_samples;
static constexpr int SAMPLE_COUNT = 4;
static constexpr int BufferSize = 4; // 버퍼 크기
std::deque<cv::Mat> frameBuf;

bool computeTopFaceDimensions_ChullPCA(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    float& width_mm, float& depth_mm, float& height_mm,
    pcl::visualization::PCLVisualizer::Ptr viewer = nullptr)
{
    if (!cloud || cloud->empty()) return false;


    // -----------------[세그멘테이션]------------------

    // 1) 윗면 평면 분할
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.06); // 복셀로 최대 5cm 간격 떨어트릴태니 그 외 포인트밀도 제외시키기
    seg.setEpsAngle(15.0f * M_PI / 180.0f);
    seg.setInputCloud(cloud);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
    seg.segment(*inliers, *coeff);
    if (inliers->indices.size() < 100) return false;

    pcl::PointCloud<pcl::PointXYZI>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*plane_cloud);
    if (plane_cloud->empty()) return false;

    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(plane_cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(0.06);         // 클러스터 간 거리 기준 (2cm)
    ec.setMinClusterSize(100);            // 너무 작은 클러스터 제거
    ec.setMaxClusterSize(10000);         // 너무 큰 클러스터 제거
    ec.setSearchMethod(tree);
    ec.setInputCloud(plane_cloud);
    ec.extract(cluster_indices);

    if (cluster_indices.empty()) return false;

    // 가장 큰 클러스터만 사용
    auto& largest_cluster = *std::max_element(cluster_indices.begin(), cluster_indices.end(),
        [](const pcl::PointIndices& a, const pcl::PointIndices& b) {
        return a.indices.size() < b.indices.size();
    });

    pcl::PointCloud<pcl::PointXYZI>::Ptr cleaned_plane(new pcl::PointCloud<pcl::PointXYZI>);
    for (int idx : largest_cluster.indices) {
        cleaned_plane->push_back((*plane_cloud)[idx]);
    }
    plane_cloud = cleaned_plane;

    // ----------------[세그멘테이션 끝]-------------


    // ------------[전처리]------------

    struct FilterParams {
        float voxel_leaf;
        int sor_mean_k;
        float sor_stddev;
        float ror_radius;
        int ror_min_neigh;
    };

	size_t N = plane_cloud->size();
    std::ostringstream oss;
    oss << "[pln] Points:       " << plane_cloud->size() << " pts\n";

    FilterParams P;
    if (N < 500) {
        P = { 0.005f, 35, 1.0f, 0.05f, 5 };
    }
    else if (N < 3000) {
        P = { 0.01f, 50, 1.5f, 0.10f, 3 };
    }
    else {
        P = { 0.02f, 0, 0.0f, 0.00f, 0 };
    }

    pcl::VoxelGrid<pcl::PointXYZI> vg;
    vg.setInputCloud(plane_cloud);
    vg.setLeafSize(0.05f, P.voxel_leaf, P.voxel_leaf);
    vg.filter(*plane_cloud);

    //pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
    //sor.setInputCloud(plane_cloud);
    //sor.setMeanK(P.sor_mean_k);
    //sor.setStddevMulThresh(P.sor_stddev);
    //sor.filter(*plane_cloud);

    //pcl::RadiusOutlierRemoval<pcl::PointXYZI> ror;
    //ror.setInputCloud(plane_cloud);
    //ror.setRadiusSearch(P.ror_radius);
    //ror.setMinNeighborsInRadius(P.ror_min_neigh);
    //ror.filter(*plane_cloud);

    // plane_cloud 시각화
	if (viewer) {
		viewer->removePointCloud("plane_cloud");
		viewer->addPointCloud<pcl::PointXYZI>(plane_cloud, "plane_cloud");
		viewer->setPointCloudRenderingProperties(
			pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "plane_cloud");
	}

    oss << std::fixed << std::setprecision(2)
        << "[vg]  leaf:          " << P.voxel_leaf * 100 << " cm\n"
		<< "[sor] Mean K:    " << P.sor_mean_k << "\n"
		<< "[sor] Stddev:     " << P.sor_stddev << "\n"
		<< "[ror] Radius:      " << P.ror_radius * 100 << " cm\n"
		<< "[ror] Neigh:       " << P.ror_min_neigh << " ea";

    viewer->removeShape("Params_text");
    viewer->addText(
        oss.str(),
        10, 200, 14, 1.0 ,1.0 ,1.0,
        "Params_text"
    );

    // --------------[전처리 끝]--------------



    // -------------[2d 평면, chull 외각선]---------------

    // 2) plane_cloud -> 2D 포인트 (y,z) 평면 투영
    pcl::PointCloud<pcl::PointXYZ>::Ptr pts2d(new pcl::PointCloud<pcl::PointXYZ>);
    pts2d->reserve(plane_cloud->size());
    for (auto& p3 : plane_cloud->points) { // p3: 평면 클러스터 클라우드(y깊이 z너비), p2: 바운딩박스(x깊이 y너비)
        pcl::PointXYZ p2;
        p2.x = p3.y; 
        p2.y = p3.z;
		p2.z = 0.0f; // z는 0으로 설정
        pts2d->push_back(p2);
    }

    // 3) Convex Hull 으로 외곽 다각형 구하기
    pcl::ConvexHull<pcl::PointXYZ> hull;
    hull.setDimension(2);
    hull.setInputCloud(pts2d); // pts2d=plane_cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr hull_pts(new pcl::PointCloud<pcl::PointXYZ>);
    hull.reconstruct(*hull_pts);
    if (hull_pts->size() < 3) return false;

    // -----------------------------------------------

    float minY = FLT_MAX, maxY = -FLT_MAX,
        minZ = FLT_MAX, maxZ = -FLT_MAX;
    for (auto& p : *pts2d) {
        minY = std::min(minY, p.x);
        maxY = std::max(maxY, p.x);
        minZ = std::min(minZ, p.y);
        maxZ = std::max(maxZ, p.y);
    }

    // 해상도: 512×512 (조절 가능)
    int W = 512, H = 512;
    cv::Mat raw = cv::Mat::zeros(H, W, CV_8UC1);
    for (auto& p : *pts2d) {
        int u = int((p.x - minY) / (maxY - minY) * (W - 1));
        int v = int((p.y - minZ) / (maxZ - minZ) * (H - 1));
        raw.at<uchar>(H - 1 - v, u) = 255;  // v축 뒤집어서 그리기
    }
	
	frameBuf.push_back(raw.clone());
	if (frameBuf.size() > BufferSize) {
		frameBuf.pop_front();
	}

    cv::Mat acc = cv::Mat::zeros(raw.size(), CV_32F);
    for (auto& f : frameBuf) {
        cv::Mat tmp;
        f.convertTo(tmp, CV_32F, 1.0 / 255.0);
        acc += tmp;
    }
    cv::Mat acc8;
    acc.convertTo(acc8, CV_8U, 255.0f / float(frameBuf.size()));

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, { 15,15 });
    cv::morphologyEx(acc8, acc8, cv::MORPH_CLOSE, kernel);

	cv::imshow("1) Raw", raw);
	cv::waitKey(1);

	cv::imshow("1) Accumulated", acc8);
	cv::waitKey(1);

    std::vector<cv::Vec3f> houghCircles;
    cv::HoughCircles(
        acc8,
        houghCircles,
        cv::HOUGH_GRADIENT,
        1.0,              // dp
        90,            // minDist between centers
        100, 50,        // param1=Canny thresh, param2=accumulator thresh
        100,   // minRadius (0.1m → 픽셀 스케일로 변환)
        250    // maxRadius (0.5m → 픽셀 스케일)
    );

    cv::Mat dbg;
	cv::cvtColor(acc8, dbg, cv::COLOR_GRAY2BGR);
    for (auto& c : houghCircles) {
		cv::Point ctr(cvRound(c[0]), cvRound(c[1]));
		int r = cvRound(c[2]);
		cv::circle(dbg, ctr, r, cv::Scalar(0, 255, 0), 2); // 원 그리기
		cv::circle(dbg, ctr, 2, cv::Scalar(0, 0, 255), -1); // 중심점 그리기
    }
	cv::imshow("2) Hough Circles", dbg);
	cv::waitKey(1);

    // --- 3) 검출된 원들을 3D 뷰어에 그리기 ---
    float plane_x = -coeff->values[3] / coeff->values[0] + 0.005f;
    for (size_t i = 0; i < houghCircles.size(); ++i) {
        float uc = houghCircles[i][0], vc = houghCircles[i][1], rp = houghCircles[i][2];
        // 픽셀→실제 y,z 좌표 역변환
        float cy = minY + uc / (W - 1) * (maxY - minY);
        float cz = minZ + (H - 1 - vc) / (H - 1) * (maxZ - minZ);
        float r = rp / (W - 1) * (maxY - minY);  // m 단위 반지름

        // 반투명 구체로 원 모델
        std::string id = "hough_circle_" + std::to_string(i);
        viewer->addSphere(
            pcl::PointXYZ(plane_x, cy, cz),
            r,
            0.0, 0.0, 1.0,
            id
        );
        viewer->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, id);

    }

    // 원 검출 성공 여부
    bool found = !houghCircles.empty();
    if (found) {
        float r = houghCircles[0][2] / (W - 1) * (maxY - minY);
        width_mm = depth_mm = 2 * r * 1000.0f;
    }
    else {
        std::vector<cv::Point2f> cv_pts;
        cv_pts.reserve(hull_pts->size());
        for (auto& p : hull_pts->points) { cv_pts.emplace_back(p.x, p.y); }

        cv::RotatedRect r = cv::minAreaRect(cv_pts); // 매핑

        float angle_deg = r.angle;

        viewer->removeShape("angle_text");
        viewer->addText("Angle: " + std::to_string(angle_deg) + u8" °", 10, 80, 14, 1.0, 1.0, 1.0, "angle_text");

        cv::Point2f corner[4];
        r.points(corner);

        if (std::abs(angle_deg) < 45.0f) {
            depth_mm = r.size.width * 1000.0f;
            width_mm = r.size.height * 1000.0f;
        }
        else {
            width_mm = r.size.width * 1000.0f;
            depth_mm = r.size.height * 1000.0f;
        }
        float plane_x = -coeff->values[3] / coeff->values[0] + 0.005f;


        // 4개 모서리 선 그리기
        for (int i = 0; i < 4; ++i) {
            const auto& A = corner[i], & B = corner[(i + 1) % 4];
            std::string id = "obb_edge_" + std::to_string(i);
            viewer->addLine(
                pcl::PointXYZ(plane_x, A.x, A.y),
                pcl::PointXYZ(plane_x, B.x, B.y),
                0.0, 1.0, 0.0, id);
            viewer->setShapeRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, id);
        }
    }


    float max_x = -std::numeric_limits<float>::infinity();
    for (auto& pt : plane_cloud->points)
        max_x = std::max(max_x, pt.x);
    float ground_offset_mm = fixed_ground_height * 1000.0f;
    height_mm = max_x * 1000.0f + ground_offset_mm;

    if (viewer) {       
        // 결과 텍스트
        viewer->removeShape("dim_text");
        std::ostringstream oss;
        oss << "W: " << width_mm  << " mm\n"
            << "D: " << depth_mm  << " mm\n"
            << "H: " << height_mm << " mm";

        viewer->addText(oss.str(), 10, 100, 16, 1,1,1, "dim_text");
    }

    return true;
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








bool accumulatePlaneClusters(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_in,
    float roiXmin, float roiXmax,
    float roiYmin, float roiYmax,
    float roiZmin, float roiZmax,
    int   bufferSize,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& outMerged,
    bool resetAccum,
    pcl::visualization::PCLVisualizer::Ptr viewer = nullptr)
{
    using clk = std::chrono::high_resolution_clock;

    auto t0 = clk::now();
    // 1) ROI 내 점만 필터링
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_roi(new pcl::PointCloud<pcl::PointXYZI>);
    for (const auto& pt : cloud_in->points) {
        if (pt.x >= roiXmin && pt.x <= roiXmax &&
            pt.y >= roiYmin && pt.y <= roiYmax &&
            pt.z >= roiZmin && pt.z <= roiZmax) {
            cloud_roi->push_back(pt);
        }
    }
    if (cloud_roi->empty()) return false;

    auto t1 = clk::now();

    // 2) 밀집 클러스터만 추출 (Euclidean Clustering)
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(cloud_roi);
    std::vector<pcl::PointIndices> clusters;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(0.04f);      // 5cm 이내 점들만 묶음
    ec.setMinClusterSize(200);          // 최소 100점 이상
    ec.setMaxClusterSize(100000);      // 최대 제한은 크게
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_roi);
    ec.extract(clusters);
    if (clusters.empty()) return false;

    // 예: 가장 큰 군집 하나만 쓸 경우
    auto largest = *std::max_element(clusters.begin(), clusters.end(),
        [](auto& a, auto& b) { return a.indices.size() < b.indices.size(); });
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_dense(new pcl::PointCloud<pcl::PointXYZI>);
    {
        pcl::ExtractIndices<pcl::PointXYZI> ex;
        pcl::PointIndices::Ptr idx(new pcl::PointIndices(largest));
        ex.setInputCloud(cloud_roi);
        ex.setIndices(idx);
        ex.setNegative(false);
        ex.filter(*cloud_dense);
    }

    auto t2 = clk::now();

    // 3) 단일 RANSAC 평면 추출
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.08f);
    seg.setMaxIterations(2000000);
    seg.setProbability(0.90f);
    seg.setInputCloud(cloud_dense);
    seg.segment(*inliers, *coeff);
    if (inliers->indices.size() < 100) return false;

    // 4) 추출된 평면만 뽑아서 merged로 사용
    pcl::PointCloud<pcl::PointXYZI>::Ptr plane(new pcl::PointCloud<pcl::PointXYZI>());
    {
        pcl::ExtractIndices<pcl::PointXYZI> ex;
        ex.setInputCloud(cloud_dense);
        ex.setIndices(inliers);
        ex.setNegative(false);
        ex.filter(*plane);
    }

    auto t3 = clk::now();

    static std::deque<pcl::PointCloud<pcl::PointXYZI>::Ptr> buf;
    if (resetAccum) buf.clear();

    // 3) 추출된 모든 평면을 전역 버퍼에 누적
    buf.push_back(plane);
    if ((int)buf.size() > bufferSize) buf.pop_front();
    //std::cerr << "[DBG] buf size = " << buf.size() << "\n";


    // 4) 누적 상태 표시
    if (viewer) {
        viewer->removeShape("status_text");
        if ((int)buf.size() < bufferSize) {
            viewer->addText(
                "Accumulating: " + std::to_string(buf.size()) + "/" + std::to_string(bufferSize),
                10, 550, 12, 1, 1, 1, "status_text");
        }
        else {
            viewer->addText(
                "Ready to measure!",
                10, 550, 12, 0.5, 0.8, 1.0, "status_text");
        }
    }

    if ((int)buf.size() < bufferSize)
        return false;

    pcl::PointCloud<pcl::PointXYZI>::Ptr merged(new pcl::PointCloud<pcl::PointXYZI>());
    for (auto& f : buf) *merged += *f;
    outMerged = merged;

    auto t4 = clk::now();

    std::ostringstream oss;
	oss << "(1) Accumlata Times\n"
        << "- ROI filtering: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms,\n"
        << "- Clustering: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms,\n"
        << "- Segmentation: " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << "ms,\n"
        << "- Buffer merged: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << "ms\n"
        << " (All Times: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4-t0).count() << "ms)";
    viewer->removeShape("1clk_delay");
	viewer->addText(
		oss.str(),
		750, 425, 12, 1.0, 1.0, 1.0,
		"1clk_delay"
	);
    
    return true;
}








void filterMergedCloud(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_in,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_out,
    pcl::visualization::PCLVisualizer::Ptr viewer = nullptr,
    bool reach_off_counter=false,
    float tmp_height_mm = 0.0f)
{
	using clk = std::chrono::high_resolution_clock;
	auto t0 = clk::now();

    // 1) 포인트 수에 따른 파라미터 결정
    struct FilterParams {
        float voxel_leaf;
        int sor_mean_k;
        float sor_stddev;
    } P;
    size_t N = cloud_in->size();

    if (!reach_off_counter) { // 카운터일때
        if (N < 1000) {
            P = { 0.025f, 60, 0.5f }; // 2st_1 | height=244.9
        }
        else if (N < 1500) {
            P = { 0.01f, 10, 0.25 }; // 2st_2 | height=268.3
        }
        else if (N < 2000) {
            P = { 0.005f, 80, 1.5f };
        }
        else if (N < 2500) {
            P = { 0.03f, 10, 0.5f }; // 2st_3 | height=260 && 2st_4 | height=
        }
        else if (N < 3000) {
            P = { 0.02f, 65, 0.25f };
        }
        else if (N < 5000) {
            P = { 0.02f, 65, 0.25f };
        }
        else if (N < 7000) {
            if (tmp_height_mm < 1200) P = { 0.02f, 65, 0.25f };
            else P = { 0.005f, 75, 1.75f }; // 2st_7
        }
        else if (N < 9000) {
            if (tmp_height_mm < 1200) P = { 0.02f, 65, 0.25f };
            else P = { 0.01f, 90, 2.75 }; // 2st_6 *
        }
        else if (N < 10000) {
            if (tmp_height_mm < 1400) P = { 0.015f, 10, 0.5f }; // 2st_11
            else P = { 0.03f, 30, 2.25f }; // 2st_13 , 2st_15
        }
        else if (N < 11000) {
            P = { 0.015, 10, 0.5 }; // 2st_11
        }
        else if (N < 12000) {
            if (tmp_height_mm < 1400) P = { 0.01f, 90, 1.25 }; // 2st_10 *
            else P = { 0.01f, 40, 0.25f }; //
        }
        else if (N < 13000) {
            P = { 0.025f, 20, 0.25f };
        }
        else if (N < 15000) {
            P = { 0.02f, 95, 2.75f }; // 2st_12 
        }
        else if (N < 17000) {
            P = { 0.005f, 80, 2.0f }; // 2st_14 
        }
        else if (N < 18000) {
            P = { 0.005f, 75, 1.5f };
        }
        else if (N < 20000) {
            P = { 0.005f, 65, 1.0f };
        }
        else if (N < 24000) {
            P = { 0.015, 20, 0.5 };
        }
        else if (N < 25000) {
            P = { 0.015, 25, 1.25 };
        }
        else if (N < 27000) {
            P = { 0.003f, 10, 1.5f };
        }
        else if (N < 55000) {
            P = { 0.01f, 50, 1.5f };
        }
        else {
            P = { 0.02f, 65, 1.7f };
        } // 0.02 10 0.25 2st_15
        // 0.02 10 2.25 2st_6 1275h
    }
    else { // 리치 지게차
        if (N < 2000) {
            P = { 0.015f, 100, 0.75f }; //   ㅇ
        }
        else if (N < 4000) {
            P = { 0.02f, 10, 0.25f }; // h 385 box 2개 ㅇ
        }
        else if (N < 10000) {
            if (tmp_height_mm < 700) P = { 0.01f,95,1.25f }; // miniboxs   ㅇ
            else P = { 0.025f, 25, 1.25f }; // h 780   ㅇ
        }
        else if (N < 14000) {
            if (tmp_height_mm < 700) P = { 0.01f,95,1.25f }; 
            else P = { 0.03f, 15, 1.5f }; // h 905    ㅇ
        }
        else if (N < 20000) {
            if (tmp_height_mm < 1000) P = { 0.02f, 30, 1.25f }; // h 870   ㅇ
            else P = { 0.02f, 70, 1.25f }; // h 1180~1240
        }
        else if (N < 22000) {
            if (tmp_height_mm < 1000) P = { 0.02f, 30, 1.25f }; // h 870
            else P = { 0.02f, 80, 1.5f }; // h 1180~1240   X
        }
        else if (N < 60000) {
            if (tmp_height_mm < 1300) P = { 0.015f, 20, 0.5f }; // h 1180~1240
            else P = { 0.01f, 50, 1.0f }; // h 1400
        }
        else if (N < 70000) {            
            P = { 0.01f, 40, 0.75 }; // h 1250
        }
        else if (N < 80000) {            
            P = { 0.02f, 35, 1.0f }; // h 1650
        }
        else if (N < 90000) {
            P = { 0.02f, 15, 0.75 }; // h 1595, h 1610
        }
        else if (N < 100000) {
            if (tmp_height_mm < 1800) P = { 0.03f, 15, 2.25 }; // h 1600
            else P = { 0.03f, 5, 0.25f }; // h 1890
        }
        else {
            P = { 0.03f, 5, 0.25f };
        }
    }

    // 2) VoxelGrid: 다운샘플링
    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    voxel.setInputCloud(cloud_in);
    voxel.setLeafSize(P.voxel_leaf, P.voxel_leaf, P.voxel_leaf);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_voxel(new pcl::PointCloud<pcl::PointXYZI>());
    voxel.filter(*cloud_voxel);

    // 3) StatisticalOutlierRemoval: 노이즈 제거
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
    sor.setInputCloud(cloud_voxel);
    sor.setMeanK(P.sor_mean_k);
    sor.setStddevMulThresh(P.sor_stddev);
    cloud_out.reset(new pcl::PointCloud<pcl::PointXYZI>());
    sor.filter(*cloud_out);


    std::ostringstream oss;
    oss << "[pln]  inPoints:       " << cloud_in->size() << " pts\n";
    oss << "[pln] outPoints:       " << cloud_out->size() << " pts\n";

    if (viewer) {
        viewer->removePointCloud("cloud_out");
        viewer->addPointCloud<pcl::PointXYZI>(cloud_in, "cloud_out");
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_out");
    }

    oss << std::fixed << std::setprecision(2)
        << "[vg]  leaf:          " << P.voxel_leaf * 100 << " cm\n"
        << "[sor] Mean K:    " << P.sor_mean_k << "\n"
        << "[sor] Stddev:     " << P.sor_stddev << "\n";


    viewer->removeShape("Params_text");
    viewer->addText(
        oss.str(),
        10, 200, 14, 1.0, 1.0, 1.0,
        "Params_text"
    );

	auto t1 = clk::now();
	std::ostringstream clk_oss;
    clk_oss << "(2) Filter Times:\n"
        << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms";
	viewer->removeShape("2clk_delay");
	viewer->addText(
		clk_oss.str(),
		750, 400, 12, 1.0, 1.0, 1.0,
		"2clk_delay"
	);
}

struct FilterParams {
    float voxel_leaf;   // VoxelGrid leaf size (m)
    int   sor_mean_k;   // SOR mean K
    float sor_stddev;   // SOR stddev multiplier
};
void filterMergedCloudWithParams(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_in,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_out,
    const FilterParams& P)
{
    // 1) VoxelGrid 다운샘플링
    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    voxel.setInputCloud(cloud_in);
    voxel.setLeafSize(P.voxel_leaf, P.voxel_leaf, P.voxel_leaf);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_voxel(new pcl::PointCloud<pcl::PointXYZI>());
    voxel.filter(*cloud_voxel);

    // 2) StatisticalOutlierRemoval
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
    sor.setInputCloud(cloud_voxel);
    sor.setMeanK(P.sor_mean_k);
    sor.setStddevMulThresh(P.sor_stddev);
    cloud_out.reset(new pcl::PointCloud<pcl::PointXYZI>());
    sor.filter(*cloud_out);
}

bool pixelizeAndDetectCircles(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    float minY, float maxY,
    float minZ, float maxZ,
    float meterPerPixel,
    std::vector<cv::Vec3f>& outCircles,
    pcl::visualization::PCLVisualizer::Ptr viewer = nullptr)
{
	using clk = std::chrono::high_resolution_clock;
	auto t0 = clk::now();

    // 1) 영상 크기 계산
    int imgW = static_cast<int>(std::ceil((maxY - minY) / meterPerPixel));
    int imgH = static_cast<int>(std::ceil((maxZ - minZ) / meterPerPixel));
    if (imgW <= 0 || imgH <= 0) return false;

    // 2) 2D 영상 생성
    cv::Mat img = cv::Mat::zeros(imgH, imgW, CV_8UC1);
    for (const auto& p : cloud->points) {
        int u = static_cast<int>((p.y - minY) / meterPerPixel);
        int v = static_cast<int>((p.z - minZ) / meterPerPixel);
        if (u < 0 || u >= imgW || v < 0 || v >= imgH) continue;
        img.at<uchar>(imgH - 1 - v, u) = 255;
    }

	cv::Mat mop;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(21,21));
    //cv::morphologyEx(img, mop, cv::MORPH_OPEN, kernel);
	cv::morphologyEx(img, mop, cv::MORPH_CLOSE, kernel);

    //cv::imshow("mop", mop);


    int minRadius = int(0.20f / meterPerPixel);  // 40 (반지름20cm)
    int maxRadius = int(0.60f / meterPerPixel);  // 120 (반지름 60

    // 4) Hough Circle 검출
    cv::HoughCircles(
        mop,
        outCircles,
        cv::HOUGH_GRADIENT,
        1.0,
        minRadius * 1.5,  // 최소 중심 간 거리
        100,         // Canny 상위 임계값
        20,          // 원 검출 임계값
        minRadius,  // 최소 반지름 (m->px)
        maxRadius    // 최대 반지름
    );

    // 5) 결과 표시
    cv::Mat color;
    cv::cvtColor(img, color, cv::COLOR_GRAY2BGR);
    for (const auto& c : outCircles) {
        cv::Point center(cvRound(c[0]), cvRound(c[1]));
        int radius = cvRound(c[2]);
        cv::circle(color, center, radius, cv::Scalar(0, 255, 0), 2);
        cv::circle(color, center, 2, cv::Scalar(0, 0, 255), -1);
    }
    //cv::imshow("Detected Circles", color);
    //cv::waitKey(1);

	auto t1 = clk::now();
	std::ostringstream oss;
	oss << "(3) Circle Detection Times:\n"
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms";
	viewer->removeShape("3clk_delay");
	viewer->addText(
		oss.str(),
		750, 375, 12, 1.0, 1.0, 1.0,
		"3clk_delay"
	);


	// 6) 원 검출 결과 반환
	if (outCircles.empty()) return false;
	return true;
}


float MeasureHeight(const pcl::PointCloud<pcl::PointXYZI>::Ptr& Load_cloud) {
    if (!Load_cloud || Load_cloud->empty()) return -1.0f;


    // 2) 밀집 클러스터만 추출 (Euclidean Clustering)
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(Load_cloud);
    std::vector<pcl::PointIndices> clusters;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(0.04f);      // 5cm 이내 점들만 묶음
    ec.setMinClusterSize(30);          // 최소 100점 이상
    ec.setMaxClusterSize(100000);      // 최대 제한은 크게
    ec.setSearchMethod(tree);
    ec.setInputCloud(Load_cloud);
    ec.extract(clusters);
    if (clusters.empty()) return false;

    // 예: 가장 큰 군집 하나만 쓸 경우
    auto largest = *std::max_element(clusters.begin(), clusters.end(),
        [](auto& a, auto& b) { return a.indices.size() < b.indices.size(); });
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_dense(new pcl::PointCloud<pcl::PointXYZI>);
    {
        pcl::ExtractIndices<pcl::PointXYZI> ex;
        pcl::PointIndices::Ptr idx(new pcl::PointIndices(largest));
        ex.setInputCloud(Load_cloud);
        ex.setIndices(idx);
        ex.setNegative(false);
        ex.filter(*cloud_dense);
    }
    float max_x = -std::numeric_limits<float>::infinity();
    for (auto& pt : Load_cloud->points) max_x = std::max(max_x, pt.x);
    return max_x * 1000.0f + fixed_ground_height * 1000.0f;    
}

bool computeDimensionsFromPlane(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& plane_cloud,
    bool circleDetected,
    float& width_mm, float& depth_mm, float& height_mm,
    pcl::visualization::PCLVisualizer::Ptr viewer = nullptr)
{
	using clk = std::chrono::high_resolution_clock;
	auto t0 = clk::now();

	if (!plane_cloud || plane_cloud->empty()) return false;

    if (viewer) {
        for (int i = 0; i < 4; ++i) {
            viewer->removeShape("rect_edge" + std::to_string(i));
            viewer->removeShape("obb_edge" + std::to_string(i));
        }
        viewer->removeShape("angle_text");
        viewer->removeShape("dim_text");
    } 

    if (circleDetected) {
        float min_y = std::numeric_limits<float>::max(), max_y = -min_y;
        float min_z = std::numeric_limits<float>::max(), max_z = -min_z;
        for (auto& p : plane_cloud->points) {
            min_y = std::min(min_y, p.y);
            max_y = std::max(max_y, p.y);
            min_z = std::min(min_z, p.z);
            max_z = std::max(max_z, p.z);
        }
        depth_mm = (max_y - min_y) * 1000.0f; // mm 단위
        width_mm = (max_z - min_z) * 1000.0f; // mm 단위

		
        if (viewer) {
            float plane_x = plane_cloud->points[0].x + 0.05f;
            pcl::PointXYZ A(plane_x, min_y, min_z);
            pcl::PointXYZ B(plane_x, max_y, min_z);
            pcl::PointXYZ C(plane_x, max_y, max_z);
            pcl::PointXYZ D(plane_x, min_y, max_z);
            viewer->addLine(A, B, 1.0, 0.0, 0.0, "rect_edge0");
            viewer->addLine(B, C, 1.0, 0.0, 0.0, "rect_edge1");
            viewer->addLine(C, D, 1.0, 0.0, 0.0, "rect_edge2");
            viewer->addLine(D, A, 1.0, 0.0, 0.0, "rect_edge3");
        }
    }
    else {
        // Convex Hull + OBB
        pcl::PointCloud<pcl::PointXYZ>::Ptr pts2d(new pcl::PointCloud<pcl::PointXYZ>());
        for (auto& p3 : plane_cloud->points) {
            pts2d->push_back({ p3.y, p3.z, 0.0f });
        }
        pcl::ConvexHull<pcl::PointXYZ> hull;
        hull.setDimension(2);
        hull.setInputCloud(pts2d);
        pcl::PointCloud<pcl::PointXYZ>::Ptr hull_pts(new pcl::PointCloud<pcl::PointXYZ>());
        hull.reconstruct(*hull_pts);
        if (hull_pts->size() < 3) return false;

        std::vector<cv::Point2f> pts;
        for (auto& p : hull_pts->points) pts.emplace_back(p.x, p.y);
        cv::RotatedRect r = cv::minAreaRect(pts);
        float angle_deg = r.angle;
        cv::Point2f corner[4];
        r.points(corner);
        if (viewer) {
            float plane_x = plane_cloud->points[0].x + 0.05f;
            for (int i = 0; i < 4; ++i) {
                auto A = corner[i], B = corner[(i + 1) % 4];
                viewer->addLine(
                    pcl::PointXYZ(plane_x, A.x, A.y),
                    pcl::PointXYZ(plane_x, B.x, B.y),
                    0, 1, 0, "obb_edge" + std::to_string(i)
                );
            }
            viewer->addText("Angle: " + std::to_string(angle_deg) + u8" °", 10, 80, 14, 1, 1, 1, "angle_text");
        }
        if (std::abs(angle_deg) < 45) {
            depth_mm = r.size.width * 1000.0f;
            width_mm = r.size.height * 1000.0f;
        }
        else {
            width_mm = r.size.width * 1000.0f;
            depth_mm = r.size.height * 1000.0f;
        }
    }

    height_mm = MeasureHeight(plane_cloud);

    if (viewer) {
        // 결과 텍스트
        viewer->removeShape("dim_text");
        std::ostringstream oss;
        oss << "W: " << width_mm << " mm\n"
            << "D: " << depth_mm << " mm\n"
            << "H: " << height_mm << " mm";
        viewer->addText(oss.str(), 10, 100, 16, 1, 1, 1, "dim_text");
    }

	auto t1 = clk::now();
	std::ostringstream oss;
	oss << "(4) Dimension Times:\n"
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms";
	if (viewer) {
		viewer->removeShape("4clk_delay");
		viewer->addText(
			oss.str(),
			750, 350, 12, 1.0, 1.0, 1.0,
			"4clk_delay"
		);
	}
	// 결과 반환
	if (width_mm <= 0 || depth_mm <= 0 || height_mm <= 0) return false;

	return true;
}






std::atomic<bool> tuningDone{ false };
std::atomic<bool> tuningStarted{ false };
std::atomic<size_t> tuningProgress{ 0 };    // 지금까지 처리한 조합 수
size_t              tuningTotal = 1;      // 전체 조합 수
std::atomic<float>  bestLeaf{ 0.01f };
std::atomic<int>    bestMeanK{ 50 };
std::atomic<float>  bestStdDev{ 1.5f }; 
std::atomic<float>  bestWpct{ std::numeric_limits<float>::infinity() };
std::atomic<float>  bestDpct{ std::numeric_limits<float>::infinity() };
// 튜닝용
void runTuningAsync(
    pcl::PointCloud<pcl::PointXYZI>::Ptr merged,
    float real_w, float real_d, float tol_pct,
    const std::vector<float>& leafSizes,
    const std::vector<int>& sorKs,
    const std::vector<float>& sorThs)
{
    float localBestErrPct = std::numeric_limits<float>::infinity();
    float localBestWpct = std::numeric_limits<float>::infinity();
    float localBestDpct = std::numeric_limits<float>::infinity();
    //FilterParams localBest = bestP;

    // 전체 조합 개수 계산
    tuningTotal = leafSizes.size() * sorKs.size() * sorThs.size();
    tuningProgress = 0;

    int nL = leafSizes.size(), nK = sorKs.size(), nT = sorThs.size();

    constexpr int REPEAT = 10; // 반복횟수

#pragma omp parallel for collapse(3) schedule(dynamic)
    for (int idxL = 0; idxL < nL; ++idxL) {
        for (int idxK = 0; idxK < nK; ++idxK) {
            for (int idxT = 0; idxT < nT; ++idxT) {

                FilterParams P{
                  leafSizes[idxL],
                  sorKs[idxK],
                  sorThs[idxT]
                };

                float sumErr = 0.f, sumW = 0.f, sumD = 0.f;
                static thread_local pcl::PointCloud<pcl::PointXYZI>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZI>());
                for (int rep = 0; rep < REPEAT; ++rep) {
                    tmp->clear();
                    filterMergedCloudWithParams(merged, tmp, P);
                    float w_mm, d_mm, h_mm;
                    computeDimensionsFromPlane(tmp, false, w_mm, d_mm, h_mm, nullptr);

                    float w_pct = real_w > 0 ? fabs(w_mm - real_w) / real_w * 100.0f : FLT_MAX;
                    float d_pct = real_d > 0 ? fabs(d_mm - real_d) / real_d * 100.0f : FLT_MAX;
                    float err_pct = std::max(w_pct, d_pct);

                    sumErr += err_pct;
                    sumW += w_pct;
                    sumD += d_pct;
                }

                float avgErr = sumErr / REPEAT;
                float avgW = sumW / REPEAT;
                float avgD = sumD / REPEAT;


#pragma omp critical
                {
                    if (avgErr < localBestErrPct) {

                        localBestErrPct = avgErr;
                        localBestWpct = avgW;
                        localBestDpct = avgD;
                        //localBest = P;

                        bestWpct.store(avgW, std::memory_order_relaxed);
                        bestDpct.store(avgD, std::memory_order_relaxed);
                        bestLeaf.store(P.voxel_leaf, std::memory_order_relaxed);
                        bestMeanK.store(P.sor_mean_k, std::memory_order_relaxed);
                        bestStdDev.store(P.sor_stddev, std::memory_order_relaxed);
                    }
                }

                tuningProgress.fetch_add(1, std::memory_order_relaxed);
            }
            
        }
    }
    // 5) 전역에 최종 결과 저장
    //bestP = localBest;
    bestWpct = localBestWpct;
    bestDpct = localBestDpct;
    tuningDone = true;
}



// 매 프레임(또는 일정 간격)마다 호출
double GetCpuUsage() {
    static ULONGLONG prevIdle = 0, prevKernel = 0, prevUser = 0;
    FILETIME ftIdle, ftKernel, ftUser;
    if (!GetSystemTimes(&ftIdle, &ftKernel, &ftUser)) {
        return 0.0; // 실패 시 0% 리턴
    }
    auto toULL = [](const FILETIME& ft) {
        return (ULONGLONG(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
    };
    ULONGLONG idle = toULL(ftIdle);
    ULONGLONG kernel = toULL(ftKernel);
    ULONGLONG user = toULL(ftUser);

    // 첫 호출 때는 이전 값(prev*)이 0이므로 초기화만 하고 0%
    if (prevIdle == 0 && prevKernel == 0 && prevUser == 0) {
        prevIdle = idle;
        prevKernel = kernel;
        prevUser = user;
        return 0.0;
    }

    ULONGLONG idleDelta = idle - prevIdle;
    ULONGLONG kernelDelta = kernel - prevKernel;
    ULONGLONG userDelta = user - prevUser;
    ULONGLONG totalDelta = kernelDelta + userDelta;

    prevIdle = idle;
    prevKernel = kernel;
    prevUser = user;

    double cpuPct = totalDelta
        ? (double)(totalDelta - idleDelta) * 100.0 / double(totalDelta)
        : 0.0;

    // 0~100% 사이로 클램프
    return std::clamp(cpuPct, 0.0, 100.0);
}



static std::string NormalizePath(const std::string& p) {
    std::string out = p;
    std::replace(out.begin(), out.end(), '\\', '/');
    return out;
}


std::chrono::steady_clock::time_point last_frame_time = std::chrono::steady_clock::now();
double fps = 0.0;

std::chrono::steady_clock::time_point tuning_start;
bool tuning_timer_started = false;

std::vector<std::string> pcd_files;
size_t pcd_index = 0;

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, const char* argv[]) {
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

    vtkOutputWindow::GetInstance()->PromptUserOff();
    vtkObject::GlobalWarningDisplayOff();

    ShellExecute(NULL, "open", "updater.exe", NULL, NULL, SW_HIDE);


    std::cout << "Current working directory: "
        << std::filesystem::current_path() << std::endl;

    curl_global_init(CURL_GLOBAL_ALL); // curl 전역 초기화 (프로그램 시작 시 한번 호출)

    lastLidarTimeMillis.store(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());

    long long now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    long long last_ms = lastLidarTimeMillis.load();

    // 예: 3000ms (=3초) 이상 데이터 수신이 없으면 끊긴 것으로 본다.
    bool lidar_receiving = (now_ms - last_ms < 3000);


    // 1) 설정 로드
    const std::string LIVOX_PATH = "config/config.json";
    const std::string WATA_PATH = "config/devTEST_setting.json";
    WATAConfig config = config::ReadConfig(WATA_PATH);
	HostInfo host = config::ReadHostInfo(LIVOX_PATH);

    enableDetectPlaneYZ = config.flag_detect_plane_yz;
    enableLoadROI = config.flag_load_roi;
    enableRAWcloud = config.flag_raw_cloud;
    enableINTENSITY = config.flag_intensity;
    enableHeightOnly = config.flag_height_only;
    onReachoffCounter = config.flag_reach_off_counter;
	fixed_ground_height = onReachoffCounter ? config.reach_height/1000.0f : config.counterbalance_height/1000.0f;
    enableReplay = config.flag_replay;
    enableHeartBeat = config.flag_heart_beat;
    enableVolume = config.flag_volume;
    enableTuning = config.flag_tuning;

    bool tuneAll = config.flag_tune_all_files;

    const int   default_iteration = config.iteration;
    const int   default_mean_k = config.mean_k;
    const float default_threshold = config.threshold;
    const FovCfg default_fov0 = fov_cfg0;   // = {0,360,0,0}


    READ_PCD_FROM_FILE = config.read_file;
    bool SAVE_PCD_FROM_FILE = config.save_file;
    const std::string READ_PCD_FILE_NAME = config.read_file_name;
    const std::string SAVE_PCD_FILE_NAME = config.save_file_name;


    const int iteration = config.iteration;

    if (enableHeartBeat) {
        std::thread heartbeatThread(heartbeatThreadFunction);
        heartbeatThread.detach(); // 스레드를 분리하여 독립적으로 실행
    }


    std::regex re_wl(R"(_w(\d+)_l(\d+))");
    std::smatch m;
    float real_w = 0.0f, real_d = 0.0f;
    if (std::regex_search(READ_PCD_FILE_NAME, m, re_wl) && m.size() == 3) {
        real_w = std::stof(m[1].str());  // w 뒤 숫자
        real_d = std::stof(m[2].str());  // l 뒤 숫자
    }
    else {
        // 파싱 실패 시 기본값
        real_w = 1000.0f;
        real_d = 1000.0f;
        std::cerr << "[WARN] Filename parsing failed: " << READ_PCD_FILE_NAME << std::endl;
    }

    if (enableTuning) {
        if (tuneAll) {
            const std::string dir = config.tune_folder;
            for (auto& entry : std::filesystem::directory_iterator(dir)) {
                if (entry.path().extension() == ".pcd")
                    pcd_files.push_back(NormalizePath(entry.path().string()));
            }
            std::sort(pcd_files.begin(), pcd_files.end());
            pcd_index = 0;
        }
        else {
            pcd_files.clear();
            pcd_files.push_back(NormalizePath(config.read_file_name));
            pcd_index = 0;
        }
        READ_PCD_FROM_FILE = true;
        V_start_process = true;
        config.flag_volume = false;
        config.flag_replay = true;
    }
    if (READ_PCD_FROM_FILE) {
        const std::string& filename = pcd_files.empty()
            ? config.read_file_name    // 튜닝 꺼진 일반 읽기
            : pcd_files[pcd_index];    // 튜닝 모드 시 첫 파일
        std::cout << "Reading point cloud: " << filename << std::endl;
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(filename, *cloud_loaded) == -1) {
            PCL_ERROR("Could not read file\n");
            return -1;
        }
        std::cout << "[INFO] Loaded " << cloud_loaded->size()
            << " points from " << filename << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));

        {
            std::regex re_wl(R"(_w(\d+)_l(\d+))");
            std::smatch m;
            if (std::regex_search(READ_PCD_FILE_NAME, m, re_wl) && m.size() == 3) {
                real_w = std::stof(m[1].str());
                real_d = std::stof(m[2].str());
            }
            else {
                real_w = real_d = 1000.0f; // 기본
            }
        }
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
    auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>("Volume Measurement Program(v1.2.5)");
    /*
    [Version 정리]   
    <목표>:
	* ver 1.1. ~ : 서비스향 환경세팅(지게차 종류, 높이만/모두 측정) 간이 설정, 로직기능 버튼, 파라미터 실시간 조정, IP 정보 출력
    * ver 1.2. ~ : 신규 부피측정 알고리즘 적용, 오차 개선 및 고도화

    - version 1.1 : 초기 모델
	- version 1.1.1
        + 리치형vs카운터형 오토 환경세팅 기능 추가
			1. dev_setting.json에 "reachoffcounter": true/false 에 따라 지면 높이, 측정 범위 변경
			2. 리치형, 카운터형 각각 "reach_height", "counterbalance_height" 으로 지면 높이 .json 파일로 설정 가능
		+ 실시간 모드에서 FoV 슬라이더로 Yaw/Pitch 조정 가능
            1. 실시간 FoV 설정할 시, 라이다 설정으로 적용되어 다시 프로그램 켜도 예전 FoV 유지 (슬라이더로 새로 조절할 시 디폴트 값부터 시작)
			2. 현재로선 실시간 모드에서만 적용 가능
    - version 1.1.2
        + save 모드시 실시간 .pcd파일 생성 및 저장 (이전엔 루프 종료 후 저장, vtk 및 윈도우 종료 버튼 클릭시 루프 벗어나서 저장이 안됐었음)
        + Fov 기본값 조정 (pitch: -9~52) -> 실제론 -9 아닌 -7 적용되겠지만 sdk viewer에선 -9까지 되길래 적용시킴
    - version 1.1.3
        + dual emit 버튼 빼고 높이만 측정(height only)와 모두 측정(all) 버튼 추가
        + YZ on/off 버튼 클릭 시, 뷰어에 pcd 안보이게 수정 (속도 향상)
		+ lidar ip 및 host ip 정보 출력
    
    - version 1.2.0
		+ 신규 부피측정 알고리즘 추가(Chull, cv::minAreaRect, OBB 등)
    - version 1.2.1
        + vtk_ui, config 모듈화
    - version 1.2.2
		+ 포인트 누적 및 필터링 기능 추가 (accumulatePlaneClusters, filterMergedCloud)
		+ 원 검출 기능 개선(pixelizeAndDetectCircles)
		+ 부피 측정 기능 개선(computeDimensionsFromPlane)
    - version 1.2.3
		+ 튜닝 기능 추가 (runTuningAsync)
        + 측정 시간 표시
        + " 적재물 높이 + 점 개수 " 로 최적 파라미터 조합 전달 (다양한 형상에 대한 튜닝을 위해)
    - version 1.2.4
        + fps, cpu사용량 표시
        + 클러스터링 계산 시간 줄임 (clustertol: 0.10f -> 0.4f 정도로 낮추면 반경 좁은 영역에서 이웃 점 찾기, 넓으면 더 많은 점을 찾아버림)
        + 최적 파라미터 튜닝(랜덤 스캐닝으로 인해 최적조합이어도 랜덤 오차 발생, 최적 파라미터 조합이 10번 측정시 오차율 적을때만 결과 출력)
        + 튜닝 남은 예상 시간 출력
    - version 1.2.5
        + 최적 파라미터 튜닝 개선
			1. 기존엔 현재 .pcd파일에서 파일명의 w옆의 숫자를 실제 너비(real_width), l옆의 숫자를 실제 깊이(real_depth)로 인식했음
            2. 따라서 현재 .pcd파일에서만 최적 파라미터 조합을 추출함 + 2400개 경우의 수 조합과 10회 그 조합으로 반복해 가장 평균 오차율이 적었던 조합 추출 방식
			3. 이제는 폴더 내에 있는 모든 .pcd파일을 수동으로 수정하고 껐다 키면서 조작할 필요 없이, 전부 자동으로 최적 조합과 오차율 등을 출력한 "tuning_result.csv" 파일을 생성함



    - 
    
    
    */
    
    viewer->addCoordinateSystem(1.0);
    viewer->setBackgroundColor(0.1, 0.1, 0.1);
    viewer->setCameraPosition(4.14367, 5.29453, -3.91817, 0.946026, -0.261667, 0.191218);

    viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)viewer.get());

    x_lengths.clear();
    y_lengths.clear();
    z_lengths.clear();

    // V_start_process 상태 텍스트 초기화
    std::string initial_status = "V_start_process: " + std::string(V_start_process ? "True" : "False");
    viewer->addText(initial_status, 20, 530, 20, 1, 1, 1, "v_start_process_text");



    // 전역 변수 또는 메인 루프 상단에 추가
    bool previous_V_start_process = V_start_process;

    auto rw = viewer->getRenderWindow();
    auto iren = rw->GetInteractor();     
    
    vtk_ui::UIManager uiMgr(
        viewer,
        iren,
        config,
        fixed_ground_height,
        heightCalibration_mode,
        onReachoffCounter,
		is_paused,
        enableINTENSITY
    );
    uiMgr.Setup();


    float roiVolXMin = -fixed_ground_height + 0.18f;
    float roiVolXMax = roiVolXMin + 1.9f;
    float roiVolYMin = 0.0f;
    float roiVolYMax = 0.0f;
    float roiVolZMin = 0.0f;
    float roiVolZMax = 0.0f;
    if (config.flag_height_only) {
        roiVolYMin = config.flag_reach_off_counter ? -1.65f : -1.30f; // 카운터 깊이측정범위: 70cm
        roiVolYMax = config.flag_reach_off_counter ? -0.42f : -0.20f;
        roiVolZMin = config.flag_reach_off_counter ? -0.20f : -0.10f; // 카운터 너비측정범위: 1.1m
        roiVolZMax = config.flag_reach_off_counter ? 1.20f : 1.0f;
    }
    else {
        roiVolYMin = config.flag_reach_off_counter ? -1.65f : -1.40f;
        roiVolYMax = config.flag_reach_off_counter ? -0.42f : -0.20f;
        roiVolZMin = config.flag_reach_off_counter ? -0.20f : -0.25f;
        roiVolZMax = config.flag_reach_off_counter ? 1.20f : 1.25f;
    }




    //std::deque<pcl::PointCloud<pcl::PointXYZI>::Ptr> buf;
    float meterPerPixel = 0.005;
    int bufferSize = 4;
    pcl::PointCloud<pcl::PointXYZI>::Ptr merged(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>);






    bool exit_for_reboot = false;
    bool prev_onReach = onReachoffCounter;
    bool prev_volume = false;
	bool isConnected = false; // LiDAR 연결 상태
    bool result_shown = false; // 결과 정지시키려고
    bool mergedReady = false; // 정해진 버퍼만큼 슬라이딩 윈도우처럼 동작되도록 누적프레임을 새로운 누적프레임으로 보여지게끔

    bool needResetAccum = false;


    // 메인 루프
    while (!viewer->wasStopped() && !exit_for_reboot) {

        bool vspChanged = (V_start_process != previous_V_start_process);
        bool resetAccum = vspChanged;

        if (viewer->wasStopped()) break;

		uiMgr.Update();

        // Livox API 로 연결 상태 조회 (예시)
        uint32_t handle = g_lidar_handle.load();
        isConnected = (handle != 0);  // 혹은 API 에서 연결 상태 직접 물어보기

        // 호스트 IP는 매 loop마다 바뀔 일이 없으니 한 번만 계산해도 됩니다.
        static std::string host_ip = GetHostIP();

        std::ostringstream ss;
        ss
            << "LiDAR IP         : " << host.lidar_ip << "\n"
            << "Host IP           : " << host_ip << "\n"
			<< "Config Host   : " << host.config_host_ip << "\n"
            << "Status            : " << (lidar_receiving ? "Receiving" : "No Data");

        uiMgr.SetConnectionText(ss.str());

        try {
            viewer->spinOnce();
			if (viewer->wasStopped()) break;
            {
                // 1) FPS
                auto now = std::chrono::steady_clock::now();
                double frame_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_frame_time).count();
                last_frame_time = now;
                fps = frame_ms > 0 ? 1000.0 / frame_ms : fps;
                viewer->removeShape("perf_text");
                std::ostringstream oss_perf;
                oss_perf << "FPS: " << std::fixed << std::setprecision(1) << fps;
                viewer->addText(oss_perf.str(), 10, 10, 12, 1, 1, 1, "perf_text");

                // 2) CPU
                double cpu = GetCpuUsage();
                viewer->removeShape("cpu_text");
                std::ostringstream oss_cpu;
                oss_cpu << "CPU: " << std::fixed << std::setprecision(1) << cpu << "%";
                viewer->addText(oss_cpu.str(), 850, 700, 12, 1, 1, 1, "cpu_text");
            }


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

                size_t start_index = static_cast<size_t>(total * 0.40f);
                size_t end_index = total; // 100%

                if (reading_active) {
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
                //if (point_index >= static_cast<int>(end_index)) {
                //    if (enableReplay) {
                //        point_index = static_cast<int>(start_index); // 40% 구간으로 돌아감
                //    }
                //    else {
                //        reading_active = false;
                //    }
                //}
                if (point_index >= static_cast<int>(total)) {
                    if (enableReplay) {
                        point_index = 0;
                    }
                    else { reading_active = false; }
                }
            }
            else if (SAVE_PCD_FROM_FILE) {
                ss << "Mode: Save\n(" << SAVE_PCD_FILE_NAME << ") \n";
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
            //if (V_start_process != previous_V_start_process) {
            //    if (previous_V_start_process) {
            //        viewer->removePointCloud("cloud_pcd");
            //        viewer->removePointCloud("cloud_filtered_volume");
            //        viewer->removeShape("result");
            //        viewer->removeShape("angle_line");
            //        for (const auto& line_id : volume_line_ids) {
            //            viewer->removeShape(line_id);
            //        }
            //        volume_line_ids.clear();
            //    }
            //    viewer->removeShape("tune_stauts");
            //    viewer->removeShape("circle_status");
            //    viewer->removeShape("time_text");
            //    viewer->removeShape("tmp_height_text");
            //    viewer->removeShape("Load_points_text");
            //    viewer->removeShape("1clk_delay");
            //    viewer->removeShape("2clk_delay");
            //    viewer->removeShape("Params_text");
            //    viewer->removeShape("3clk_delay");
            //    viewer->removeShape("4clk_delay");
            //    viewer->removeShape("angle_text");
            //    viewer->removeShape("dim_text");
            //    viewer->removeShape("status_text");

            //    for (int i = 0; i < 4; ++i) {
            //        viewer->removeShape("rect_edge" + std::to_string(i));
            //        viewer->removeShape("obb_edge" + std::to_string(i));
            //    }

            //    viewer->removePointCloud("cloud_out");

            //    frameBuf.clear();

            //    uiMgr.SetDevText("");

            //    // 포인트 클라우드 데이터 초기화
            //    resetPointCloudData();

            //    // V_start_process 껐다 켜면 결과도 다시 보여주도록 플래그 리셋
            //    result_shown = false;

            //    // 이전 상태 업데이트
            //    previous_V_start_process = V_start_process;

            //    // Viewer에 현재 상태 텍스트 업데이트
            //    std::string status_text = "V_start_process: " + std::string(V_start_process ? "True" : "False");
            //    viewer->removeShape("v_start_process_text");
            //    viewer->addText(status_text, 20, 530, 20, 1, 1, 1, "v_start_process_text");

            //    std::cout << "[INFO] V_start_process state changed to "
            //        << (V_start_process ? "True" : "False") << ". Data has been reset." << std::endl;
            //}




            if (vspChanged) {

                viewer->removeShape("circle_status");
                viewer->removeShape("status_text");
                viewer->removeShape("time_text");
                viewer->removeShape("tmp_height_text");
                viewer->removeShape("Load_points_text");
                viewer->removeShape("1clk_delay");
                viewer->removeShape("2clk_delay");
                viewer->removeShape("Params_text");
                viewer->removeShape("3clk_delay");
                viewer->removeShape("4clk_delay");
                viewer->removeShape("angle_text");
                viewer->removeShape("dim_text");
                viewer->removeShape("status_text");

                for (int i = 0; i < 4; ++i) {
                    viewer->removeShape("rect_edge" + std::to_string(i));
                    viewer->removeShape("obb_edge" + std::to_string(i));
                }

                viewer->removePointCloud("cloud_out");


                uiMgr.SetDevText("");
                prev_volume = false;

                resetPointCloudData();
                merged->clear();
                frameBuf.clear();
                result_shown = false;
                needResetAccum = true;

                previous_V_start_process = V_start_process;

                std::cout << "[INFO] Accum buffer _will_ be reset on next accumulation.\n";
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
				float loadROI_y_max = config.flag_reach_off_counter ? -0.45f : -0.15f;
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


                if (count_load_roi_1 >= 10 && count_load_roi_1 <= 5000) {
                    PickUp_1 = true;
                    //std::cout << "[PICKUP] 1st Floor !!!" << std::endl;

                    std::ostringstream oss;
					oss << "Pickup 1st Floor: " << count_load_roi_1 << " points";
                    viewer->removeShape("pickup_text");
                    viewer->addText(oss.str(), 20, 350, 20, 0.0, 1.0, 0.0, "pickup_text");
                    viewer->setShapeRenderingProperties(
                        pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,
                        8, "pickup_text");

                    heightCalibration_mode = false;

                    //viewer->removeShape("load_roi_box_1");
                    //viewer->addCube(
                    //    loadROI_x1_min, loadROI_x1_max,
                    //    loadROI_y_min, loadROI_y_max,
                    //    loadROI_z_min, loadROI_z_max,
                    //    0.68, 1.0, 0.18,
                    //    "load_roi_box_1"
                    //);
                    //viewer->setShapeRenderingProperties(
                    //    pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                    //    pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
                    //    "load_roi_box_1"
                    //);
                } else
                {
                    viewer->removeShape("pickup_text");
                    viewer->removeShape("load_roi_box_1");

                    //heightCalibration_mode = true;
                }


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
                // [부피 형상 측정]
                // -------------------------------------------------------------------------------------
                if (V_start_process && PickUp_1) {
                    bool resetAccum = needResetAccum;

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
                        point.y = temp.y;
                        point.z = temp.z;
                        point.intensity = temp.intensity; // intensity 필드도 복사

                        if (point.x >= roiVolXMin && point.x < roiVolXMax &&
                            point.y >= roiVolYMin && point.y <= roiVolYMax &&
                            point.z >= roiVolZMin && point.z <= roiVolZMax)
                        {
                            x_values.push_back(point.x);
                            cloud_filtered_volume->points.push_back(point);
                            cloud_pcd_local->points.push_back(point);
                        }
                    }
                    cloud_filtered_volume->width = cloud_filtered_volume->points.size();
                    cloud_filtered_volume->height = 1;

                    pcl::VoxelGrid<pcl::PointXYZI> vg;
                    vg.setInputCloud(cloud_filtered_volume);
                    vg.setLeafSize(0.02f, 0.02f, 0.02f);  // 0.5cm 해상도
                    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ds(new pcl::PointCloud<pcl::PointXYZI>);
                    vg.filter(*cloud_ds);
                    cloud_filtered_volume.swap(cloud_filtered_volume);

                    if (config.flag_reach_off_counter) { // 리치 일때
                        //if (cloud_filtered_volume->points.size() > 2000) {
                        //    bufferSize = 3;
                        //}
                        //if (cloud_filtered_volume->points.size() > 4000) {
                        //    bufferSize = 2;
                        //}
                        if (cloud_filtered_volume->points.size() > 6000) {
                            bufferSize = 2;
                        }
                    }
                    else { // 카운터밸런스 일때
                        if (cloud_filtered_volume->points.size() > 4000) {
                            bufferSize = 2;
                        }
                        if (cloud_filtered_volume->points.size() > 9000) {
                            bufferSize = 1;
                        }
                    }
                    viewer->removeShape("Load_points_text");
                    std::ostringstream oss1;
                    oss1 << "Load points: " << cloud_filtered_volume->points.size();
                    viewer->addText(
                        oss1.str(),
                        800, 37,       // 화면 좌표
                        14,            // 글자 크기
                        1.0, 1.0, 1.0, // 흰색
                        "Load_points_text"
					);


                    if (config.flag_height_only && !config.flag_detect_plane_yz) {
                        bool orig_detect = config.flag_detect_plane_yz;
                        bool orig_volume = config.flag_volume;

                        config.flag_detect_plane_yz = false;
                        config.flag_volume = false;

                        float height_mm = MeasureHeight(cloud_filtered_volume);
                        float width_mm = 0.0f;
                        float depth_mm = 0.0f;

                        bool result_status = true;
                        std::string json_result = "{"
                            "\"height\": " + std::to_string(height_mm) + ", "
                            "\"width\": " + std::to_string(width_mm) + ", "
                            "\"length\": " + std::to_string(depth_mm) + ", "
                            "\"result\": " + std::to_string(result_status) + ", "
                            "\"timestamp\": \"" + getCurrentTime() + "\", "
                            "\"points\": [] }";

                        std::ostringstream oss;
                        oss << "Height: " << height_mm << " mm \n"
                            << "Width: " << width_mm << " mm \n"
                            << "Length: " << depth_mm << " mm \n";

                        std::string result = oss.str();
                        if (!result_shown) {
                            uiMgr.SetDevText(result);
                            result_shown = true;
                        }
                        

                        std::string msg_pub = "MID360>LIS " + json_result;
                        zmq::message_t topic_msg(msg_pub.c_str(), msg_pub.length());
                        publisher.send(topic_msg, zmq::send_flags::dontwait);

                        height_mm = 0;
                        width_mm = 0;
                        depth_mm = 0;

                        config.flag_detect_plane_yz = orig_detect;
                        config.flag_volume = orig_volume;
                    }
                    else {}




                    if (!config.flag_detect_plane_yz) {
                        viewer->removePointCloud("cloud_filtered_volume");
                    }

                    if (!x_values.empty()) {
                        calcMaxX(x_values, max_x_value);
                        for (auto& temp : cloud_filtered_volume->points) {
                            temp.x = max_x_value;
                        }
                    }



                    if (!config.flag_tuning) {
                        viewer->removeShape("tune_status");
                        viewer->removeShape("status_text");
                        viewer->removeShape("eta_text");
                        tuningStarted = tuningDone = false;
                        tuning_timer_started = false;
                    }
                    if (config.flag_tuning) {
                        static bool shownDone = false;
						float height_mm = MeasureHeight(cloud_filtered_volume);


                        if (!accumulatePlaneClusters(cloud_filtered_volume, roiVolXMin, roiVolXMax,
                            roiVolYMin, roiVolYMax, roiVolZMin, roiVolZMax,
                            bufferSize, merged, false, viewer)) {
                            viewer->spinOnce();
                            continue;
                        }

                        if (!tuningStarted) {
                            tuningStarted = true;
                            tuningDone = false;
                            shownDone = false;
                            viewer->removeShape("tune_status");
                            viewer->addText("tuning...", 200, 50, 14, 1, 1, 0, "tune_status");

                            // 튜닝 시작 시간 저장
                            tuning_timer_started = true;
                            tuning_start = std::chrono::steady_clock::now();

                            // 그리드 생성
                            std::vector<float> leafSizes;
                            for (int i = 1; i <= 6; ++i)   // 6 단계: 0.005, 0.01, …, 0.03
                                leafSizes.push_back(i * 0.005f);
                            std::vector<int> sorKs;
                            for (int k = 5; k <= 100; k += 5)
                                sorKs.push_back(k);
                            std::vector<float> sorThs;
                            for (int i = 1; i <= 20; ++i)  // 20 단계: 0.25, 0.50, …, 5.00
                                sorThs.push_back(i * 0.25f);

                            // 백그라운드 튜닝 시작
                            std::thread(runTuningAsync,
                                merged,
                                real_w, real_d, 2.0f,
                                leafSizes, sorKs, sorThs
                            ).detach();
                        }


                        // 3) 아직 튜닝 중
                        if (!tuningDone) {
                            // 진행/전체
                            size_t prog = tuningProgress.load();
                            size_t tot = tuningTotal;
                            float pct = tot ? (100.0f * prog / tot) : 0.0f;

                            // ETA 계산
                            std::string eta_str;
                            if (tuning_timer_started && prog > 0) {
                                auto now = std::chrono::steady_clock::now();
                                double elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(now - tuning_start).count();
                                double avg_per = elapsed_s / prog;
                                double remain_s = avg_per * (tot - prog);
                                int h = int(remain_s) / 3600;
                                int m = (int(remain_s) % 3600) / 60;
                                int s = int(remain_s) % 60;
                                std::ostringstream oss_eta;
                                oss_eta << "ETA: "
                                    << (h > 0 ? std::to_string(h) + "h " : "")
                                    << (m > 0 ? std::to_string(m) + "m " : "")
                                    << s << "s";
                                eta_str = oss_eta.str();
                            }

                            std::ostringstream ss;
                            ss << "Tuning: " << prog << "/" << tot
                                << " (" << std::fixed << std::setprecision(1) << pct << "%)"
                                << " | ErrW=" << std::fixed << std::setprecision(1)
                                << (bestWpct.load() == std::numeric_limits<float>::infinity() ? 0 : bestWpct.load()) << "%"
                                << ", ErrD=" << std::fixed << std::setprecision(1)
                                << (bestDpct.load() == std::numeric_limits<float>::infinity() ? 0 : bestDpct.load()) << "%"
                                << "  [leaf=" << std::fixed << std::setprecision(3) << bestLeaf.load()
                                << "  k=" << bestMeanK.load()
                                << "  std=" << std::fixed << std::setprecision(2) << bestStdDev.load()
                                << "]";
                            viewer->removeShape("tune_status");
                            viewer->addText(ss.str(), 200, 50, 14, 1, 1, 0, "tune_status");
                            // ETA 표시
                            viewer->removeShape("eta_text");
                            if (!eta_str.empty())
                                viewer->addText(eta_str, 200, 35, 12, 1, 1, 0, "eta_text");

                            if (tuneAll) {
                                size_t idx = pcd_index + 1;
                                size_t total = pcd_files.size();
                                size_t remain = total - idx;

                                std::ostringstream ossAll;
                                ossAll
                                    << "File " << idx << "/" << total
                                    << "  (Remaining: " << remain << ")";
                                viewer->removeShape("tuneAll_text");
                                viewer->removeShape("tuneAll_text");
								viewer->addText(ossAll.str(), 200, 20, 12, 1, 1, 1, "tuneAll_text");
							}
                            else {
                                viewer->removeShape("tuneAll_text");
                            }

                        }
                        // 4) 튜닝 완료
                        else {
                            if (!shownDone) {
                                // 한 번만 표시
                                std::ostringstream oss;
                                oss << "Done: leaf=" << std::fixed << std::setprecision(3) << bestLeaf.load()
                                    << ", k=" << bestMeanK.load()
                                    << ", thr=" << std::fixed << std::setprecision(2) << bestStdDev.load()
                                    << " | ErrW=" << std::fixed << std::setprecision(1) << bestWpct.load() << "%"
                                    << ", ErrD=" << std::fixed << std::setprecision(1) << bestDpct.load() << "%"
                                    << "\nHeight= " << height_mm;

                                viewer->removeShape("tune_status");
                                viewer->addText(oss.str(), 200, 50, 14, 0, 1, 0, "tune_status");
                            
                                if (tuneAll) {
                                    viewer->removeShape("tuneAll_text");
                                    viewer->addText(
                                        "Tuned: " + pcd_files[pcd_index],
                                        200, 20, 12, 1, 1, 1, "tuneAll_text"
                                    );

                                    std::string csv_path = "reach/tuning_results.csv";
                                    std::ofstream csv(csv_path, std::ios::app);
                                    bool need_header = true;
                                    if (std::filesystem::exists(csv_path) &&
                                        std::filesystem::file_size(csv_path) > 0) {
                                        need_header = false;
                                    }
                                    if (need_header) {
                                        csv << "pcd_file,leaf[m],mean_k,stddev,ErrW[%],ErrD[%],points\n";
                                    }

                                    csv
                                        << pcd_files[pcd_index] << ","
                                        << std::fixed << std::setprecision(3) << bestLeaf.load() << ","
                                        << bestMeanK.load() << ","
                                        << std::fixed << std::setprecision(2) << bestStdDev.load() << ","
                                        << bestWpct.load() << ","
                                        << bestDpct.load() << ","
                                        << cloud_filtered_volume->points.size()
                                        << "\n";
                                    csv.close();


                                    // — 다음 파일로 넘어갈 준비 —
                                    if (pcd_index + 1 < pcd_files.size()) {
                                        pcd_index++;

                                        tuningStarted = false;
                                        tuningDone = false;
                                        tuningProgress = 0;
                                        bestWpct = std::numeric_limits<float>::infinity();
                                        bestDpct = std::numeric_limits<float>::infinity();

                                        {
                                            nlohmann::json j;
                                            std::ifstream ifs(WATA_PATH);
                                            ifs >> j; ifs.close();
                                            j["file"]["read_file_name"] = pcd_files[pcd_index];
                                            std::ofstream ofs(WATA_PATH);
                                            ofs << j.dump(4) << "\n";
                                        }
                                        // 재실행 플래그 세팅 후 루프 탈출
                                        vtk_ui::rebootRequested.store(true);
                                        exit_for_reboot = true;
                                        break;  // 메인 루프 종료 -> uninit -> 재실행
                                    }
								}
                                else {
                                    viewer->removeShape("tuneAll_text");
                                }
                                
                                shownDone = true;

                            }
                        }
                    }
                    



                    if (config.flag_volume && V_start_process && !config.flag_height_only) {

                        using clk1 = std::chrono::high_resolution_clock;
						auto start = clk1::now();

						float tmp_height_mm = MeasureHeight(cloud_filtered_volume);
                        std::ostringstream ossh;
                        ossh << "height(tmp): " << tmp_height_mm << std::endl;
                        viewer->removeShape("tmp_height_text");
                        viewer->addText(ossh.str(), 800, 10, 14, 1, 1, 1, "tmp_height_text");

						mergedReady = accumulatePlaneClusters(
							cloud_filtered_volume, roiVolXMin, roiVolXMax,
							roiVolYMin, roiVolYMax, roiVolZMin, roiVolZMax,
							bufferSize, merged, resetAccum, viewer
						);
                        previous_V_start_process = V_start_process;
                        if (resetAccum) needResetAccum = false;

						if (mergedReady) {
                            filterMergedCloud(merged, filtered, viewer, config.flag_reach_off_counter, tmp_height_mm);

                            std::vector<cv::Vec3f> circles;
                            bool circleDetected = pixelizeAndDetectCircles(
                                filtered,
                                roiVolYMin, roiVolYMax,
                                roiVolZMin, roiVolZMax,
                                meterPerPixel,
                                circles, viewer
                            );

                            if (viewer) {
                                // 이전 표시 제거
                                viewer->removeShape("circle_status");
                                if (circleDetected) {
                                    viewer->addText(
                                        "Circle Detected!",
                                        10, 285,       // 화면 좌표
                                        14,            // 글자 크기
                                        0.0, 1.0, 0.0, // 초록색
                                        "circle_status"
                                    );
                                }
                                else {
                                    viewer->addText(
                                        "No Circle",
                                        10, 285,
                                        14,
                                        1.0, 0.0, 0.0, // 빨강색
                                        "circle_status"
                                    );
                                }

                            }
                            float width_mm, depth_mm, height_mm;
                            computeDimensionsFromPlane(
                                filtered,
                                circleDetected,
                                width_mm, depth_mm, height_mm,
                                viewer
                            );
                            if (height_mm != 0 && width_mm != 0 && depth_mm != 0) {
                                bool result_status = true;

                                std::string json_result = "{"
                                    "\"height\": " + std::to_string(height_mm) + ", "
                                    "\"width\": " + std::to_string(width_mm) + ", "
                                    "\"length\": " + std::to_string(depth_mm) + ", "
                                    "\"result\": " + std::to_string(result_status) + ", "
                                    "\"timestamp\": \"" + getCurrentTime() + "\", "
                                    "\"points\": [] }";

                                std::ostringstream oss;
                                oss << "Height: " << height_mm << " mm \n"
                                    << "Width: " << width_mm << " mm \n"
                                    << "Length: " << depth_mm << " mm \n";

                                std::string result = oss.str();

                                if (!result_shown) {
                                    uiMgr.SetDevText(result);
                                    result_shown = true;
                                }

                                std::string msg_pub = "MID360>LIS " + json_result;
                                zmq::message_t topic_msg(msg_pub.c_str(), msg_pub.length());
                                publisher.send(topic_msg, zmq::send_flags::dontwait);

                                height_mm = 0;
                                width_mm = 0;
                                depth_mm = 0;
                            }
                            auto end = clk1::now();

                            std::chrono::duration<double> elapsed = end - start;
                            std::ostringstream oss;
                            oss << "VolumeMeasure Time:\n"
                                << std::fixed << std::setprecision(3)
                                << elapsed.count() << " seconds";
                            //viewer->removeShape("time_text");
                            viewer->addText(
                                oss.str(),
                                750, 325,       // 화면 좌표
                                12,            // 글자 크기
                                1.0, 1.0, 0.0, // 노란색
                                "time_text"
                            );
                        }                       
                    }



                    //detectPlaneYZ(cloud_filtered_volume, viewer);
                    if (config.flag_detect_plane_yz) {
                        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> ch(
                            cloud_filtered_volume, 172, 255, 142);

                        viewer->removePointCloud("cloud_filtered_volume");
                        viewer->addPointCloud(cloud_filtered_volume, ch, "cloud_filtered_volume");

                        viewer->setPointCloudRenderingProperties(
                            pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                            2, "cloud_filtered_volume");

                        detectPlaneYZ(cloud_filtered_volume, viewer);

                        if (!config.flag_height_only) {
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
                        }
                        if (config.flag_height_only) {
                            // 높이만 측정 결과 계산
                            if (x_lengths.size() == vector_size) {
                                result_height = calculateAverageX(x_lengths) * 1000;
                                x_lengths.clear();
                            }
                        }

                        if ((result_height != 0 && result_width != 0 && result_length != 0) || (config.flag_height_only && result_height != 0 && result_width == 0 && result_length == 0)) {
                            bool result_status = true;

                            float ground_correction_mm = 0.0f;
                            ground_correction_mm = fixed_ground_height * 1000.0f;
                            result_height += (ground_correction_mm + 9.5f);

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

                            uiMgr.SetDevText(result);

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
						viewer->removePointCloud("cloud_filtered_volume");

                        //DrawOBB(cloud_filtered_volume, viewer);

					}



                }



                // PCD 데이터 저장 (단순 누적)
                if (SAVE_PCD_FROM_FILE) {
                    *cloud_pcd += *cloud_merge;
                    std::cout << "[DEBUG] Accumulated " << cloud_pcd->size() << " points in cloud_pcd. \n";

					if (pcl::io::savePCDFileBinary(SAVE_PCD_FILE_NAME, *cloud_pcd) == 0) {
						std::cout << "[INFO] Saved " << cloud_pcd->size()
							<< " points to " << SAVE_PCD_FILE_NAME << std::endl;

						std::uintmax_t bytes = std::filesystem::file_size(SAVE_PCD_FILE_NAME);
						double mb = bytes / (1024.0 * 1024.0);

                        ss << cloud_pcd->size() << " pts";
						ss << " | Size: " << std::fixed << std::setprecision(2) << mb << " MB";
					}
					else {
						PCL_ERROR("Failed to save PCD file to %s. \n", SAVE_PCD_FILE_NAME.c_str());
					}

                }

                // 파일 모드라면 다음 chunk 대기 위해 처리 종료
                if (READ_PCD_FROM_FILE) {
                    cloud_merge->clear();
                }

                uiMgr.SetStatusText(ss.str());


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


    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>[STOP DETECTION]>>>>>>>>>>>>>>>>>>>>>>>>\n";
    LivoxLidarSdkUninit();
    curl_global_cleanup();

    if (vtk_ui::rebootRequested.load()) {
#if defined(_WIN32)
        // 기존 CreateProcessA 대신 아래로 교체
        TCHAR szPath[MAX_PATH];
        GetModuleFileName(nullptr, szPath, MAX_PATH);
        ShellExecute(nullptr, TEXT("open"), szPath, nullptr, nullptr, SW_SHOWNORMAL);
#else
        // execv 는 argv[0] 과 끝에 NULL 포인터를 넘겨야 합니다.
        char* const exec_args[] = { const_cast<char*>(argv[0]), nullptr };
        execv(argv[0], exec_args);
#endif
    }

    return 0;
}
