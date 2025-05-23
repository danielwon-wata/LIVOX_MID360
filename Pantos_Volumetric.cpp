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

#include <windows.h>

#include "json.hpp" 
#include <regex>

#include <curl/curl.h>

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

#include <omp.h> // ����ó��
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

#include "resource_Pantos.h"





// ���������� LiDAR �����͸� ���� �ð� (�и���)
std::atomic<long long> lastLidarTimeMillis(0);

std::atomic<bool> heartbeatRunning(true);

// ----------------------------------------------------------------------------
// JSON ����ü
// ----------------------------------------------------------------------------
struct CaliROIBox {
    float x_min, x_max;
    float y_min, y_max;
    float z_min, z_max;
};
CaliROIBox ground_roi_box = {
    -4.0f, -2.0f,
    -0.85f, -0.35f,
    1.2f, 1.5f
    //0.2f, 0.6f // ��ũ ����
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
    // �߰�
    int mean_k = 0;
    float threshold = 0;
    int height_threshold = 0;
};





// ----------------------------------------------------------------------------
// ����
// ----------------------------------------------------------------------------
std::mutex control_mutex;
std::mutex g_mutex;
bool V_start_process = false;
bool reading_active = true; // �ʱ� ���´� �б� Ȱ��ȭ
bool is_paused = false; // �ʱ� ���´� �Ͻ����� �ƴ�

bool pallet_height_fixed = false;
float fixed_pallet_height = 0.0f;

bool heightCalibration_mode = false; // ���� Ķ���극�̼� 
float fixed_ground_height = 2.272f;
bool ground_height_fixed = false;
bool showGroundROI = false;    // ���� Ķ���극�̼ǿ� ROI �ڽ��� ǥ������ ����

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

std::vector<float> x_lengths(vector_size); // ���� ũ��� �ʱ�ȭ ��Ŵ


std::vector<std::string> volume_line_ids;

float angle_degrees = 0;
int x_index = 0;
int point_index = 0;
int pallet_line_index = 0;

float max_x_value = 0;
float result_length = 0;
float result_height = 0;
float result_width = 0;

std::ostringstream oss;

// ----------------------------------------------------------------------------
// JSON ���� �б�
// ----------------------------------------------------------------------------
using json = nlohmann::json;

std::string removeComments(const std::string& jsonString) {
    std::regex singleLineCommentRegex("//.*?$");
    std::regex multiLineCommentRegex("/\\*.*?\\*/");

    // ���� �� ���� �� �ּ� ����
    std::string withoutComments = std::regex_replace(jsonString, multiLineCommentRegex, "");
    withoutComments = std::regex_replace(withoutComments, singleLineCommentRegex, "", std::regex_constants::format_default);

    // �� ���� Ʈ���Ͽ� ����
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

    // �ּ� ����
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
// ��Ÿ ��ƿ
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

// ����Ʈ Ŭ���� ������ �ʱ�ȭ �Լ�
void resetPointCloudData() {
    std::lock_guard<std::mutex> lock(g_mutex);
    cloud_pcd->clear();
    cloud_raw->clear();
    cloud_merge->clear();
    // �ٸ� Ŭ���� �����͵� �ʱ�ȭ �ʿ� �� �߰�
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

    // ���� �ؽ�Ʈ ����
    viewer->removeShape("camera_position_text");

    // ���ο� �ؽ�Ʈ �߰�
    viewer->addText(oss.str(), 20, 700, 15, 1.0, 1.0, 1.0, "camera_position_text");
}


// ----------------------------------------------------------------------------
// ���ҽ� �ݹ� (�ǽð� ���)
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

        // ���� ����Ʈ(=96*iteration=9600)���� cloud_merge�� ����
        if (cloud_raw->points.size() >= 96 * config->iteration) {
            pcl::copyPointCloud(*cloud_raw, *cloud_merge);
            cloud_raw->clear();
        }
    }
}

// ----------------------------------------------------------------------------
// ��ó�� 1) Voxel
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
// ��ó�� 2) Outlier Remove
// ----------------------------------------------------------------------------
void removeOutliers(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const WATAConfig& config) {
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);

    sor.setMeanK(config.mean_k);
    sor.setStddevMulThresh(config.threshold);  // �� Ŀ���� ���Ű� �ٰ�, �� �۾����� ���Ű� ������

    sor.filter(*cloud);
}

// ----------------------------------------------------------------------------
// [�ڵ� ���� Ķ���극�̼�] ���� Ž��
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
    // ROI ���ڸ� ǥ�� �Ǵ� ����
    if (show) {
        viewer->removeShape("ground_roi_box");
        viewer->addCube(
            ground_roi_box.x_min, ground_roi_box.x_max,    // X ����
            ground_roi_box.y_min, ground_roi_box.y_max,  // Y ����
            ground_roi_box.z_min, ground_roi_box.z_max,     // Z ����
            1.0, 0.0, 1.0, // (R=1.0, G=1.0, B=0.0) �����
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
// Ű���� �ݹ� �Լ�
// ----------------------------------------------------------------------------
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* viewer_void) {
    auto viewer = static_cast<pcl::visualization::PCLVisualizer*>(viewer_void);

    if (event.keyDown()) {
        if (event.getKeySym() == "p" || event.getKeySym() == "P") { // 'p' Ű: �Ͻ����� ���
            std::lock_guard<std::mutex> lock(control_mutex);
            is_paused = !is_paused; // �Ͻ����� ���� ���
            if (is_paused)
                std::cout << "[INFO] Data reading paused.\n";
            else
                std::cout << "[INFO] Data reading resumed.\n";
        }
        else if (event.getKeySym() == "v" || event.getKeySym() == "V") { // 'v' Ű: V_start_process ���
            std::lock_guard<std::mutex> lock(control_mutex);
            V_start_process = !V_start_process; // V_start_process ���� ���
            if (V_start_process)
                std::cout << "[INFO] V_start_process set to true.\n";
            else
                std::cout << "[INFO] V_start_process set to false.\n";

            // ����Ʈ Ŭ���� ������ �ʱ�ȭ
            resetPointCloudData();

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
// [���� ����] ��� ���
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
// [���� ����] YZ ��� Ŭ�����͸�
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


            if (inliers->indices.size() < 100) {
                std::cout << "No more planes found." << std::endl;
                break;
            }


            // ����� ����Ʈ Ŭ���� ����
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
            extract.setInputCloud(cloud);
            extract.setIndices(inliers);
            extract.setNegative(false);
            extract.filter(*cloud_plane);

            // Ŭ�����͸�
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
// [���� ����] ���� ���
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
// [���� ����] ���� ���
// ----------------------------------------------------------------------------
void calculateAnglePoints(const pcl::PointXYZ& start_point, const pcl::PointXYZ& end_point, pcl::visualization::PCLVisualizer::Ptr viewer) {
    float dy = end_point.y - start_point.y;
    float angle_radians = std::atan(dy);  // ����


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
    CURL *curl = curl_easy_init();
    if (curl) {
        // ����𿡼� �Ҵ�� ����Ʈ
		const char* url = "http://localhost:8081/heartbeat";
        curl_easy_setopt(curl, CURLOPT_URL, url);
		curl_easy_setopt(curl, CURLOPT_POST, 1L);

        // ������ JSON ���ڿ�
        const char* postData = "{\"status\":\"alive\"}";
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postData);

        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
			std::cerr << "[ERROR] Hearbeat ���� ����: " << curl_easy_strerror(res) << std::endl;
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

		if (elapsed > 5000) { // 5�� �̻� ����ϸ�
			std::cerr << "[ERROR] LiDAR ������ ���� �ߴܵ�. ���α׷� ����." << std::endl;
			heartbeatRunning.store(false);
			break;
		}
		sendHeartbeatSignal(); // heartbeat ��ȣ ���� �Լ� ����
		std::this_thread::sleep_for(std::chrono::seconds(1)); // 1�ʸ��� ��ȣ ����
    }
}



// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, const char* argv[]) {
	// �ý��� ���� ��ȭ���� ǥ�ø� ���� (Windows ����)
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);

    std::cout << "Current working directory: "
        << std::filesystem::current_path() << std::endl;

	curl_global_init(CURL_GLOBAL_ALL); // curl ���� �ʱ�ȭ (���α׷� ���� �� �ѹ� ȣ��)

    // ���α׷� ���� ��, ������ LiDAR ���Žð� �ʱ�ȭ
    lastLidarTimeMillis.store(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());

    std::thread heartbeatThread(heartbeatThreadFunction);
	heartbeatThread.detach(); // �����带 �и��Ͽ� ���������� ����

    // 1) ���� �ε�
    const std::string LIVOX_PATH = "config/config.json";
    const std::string WATA_PATH = "config/P_setting.json";
    WATAConfig config = readConfigFromJson(WATA_PATH);

    bool READ_PCD_FROM_FILE = config.read_file;
    bool SAVE_PCD_FROM_FILE = config.save_file;
    const std::string READ_PCD_FILE_NAME = config.read_file_name;
    const std::string SAVE_PCD_FILE_NAME = config.save_file_name;


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


    if (READ_PCD_FROM_FILE) {
        // ���Ͽ��� PCD �ε�
        std::cout << "Reading point cloud: " << READ_PCD_FILE_NAME << std::endl;
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(READ_PCD_FILE_NAME, *cloud_loaded) == -1) {
            PCL_ERROR("Could not read file\n");
            return -1;
        }
        std::cout << "[INFO] Loaded " << cloud_loaded->size()
            << " points from " << READ_PCD_FILE_NAME << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    else {  // �ǽð� (Livox) ���
        try {            
            if (!LivoxLidarSdkInit(LIVOX_PATH.c_str())) {
                throw std::runtime_error("Livox Init Failed");
            }
            SetLivoxLidarPointCloudCallBack(PointCloudCallback, &config);
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
    auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>("Pantos_Volumetric_Viewer");
    viewer->addCoordinateSystem(1.0);
    viewer->setBackgroundColor(0.1, 0.1, 0.1);
    viewer->setCameraPosition(4.14367, 5.29453, -3.91817, 0.946026, -0.261667, 0.191218);

    int previous_pallet_index = 0;

    viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)viewer.get());


    x_lengths.clear();


    // V_start_process ���� �ؽ�Ʈ �ʱ�ȭ
    std::string initial_status = "V_start_process: " + std::string(V_start_process ? "True" : "False");
    viewer->addText(initial_status, 20, 350, 20, 1, 1, 1, "v_start_process_text");



    // ���� ���� �Ǵ� ���� ���� ��ܿ� �߰�
    bool previous_V_start_process = V_start_process;

    // ���� ����
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

            if (READ_PCD_FROM_FILE) { // ���Ͽ��� PCD �ε�
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

            if (heightCalibration_mode) { // ���� Ķ���극�̼� ���                
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

                try { // ���� ���� ���
                    fixed_ground_height = std::abs(calculateGroundHeight(cloud_ground));
                    ground_height_fixed = true; // ���� ���� ����(���)

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
                    ); // ���������� ���� ����Ʈ ǥ��
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
            // [ZMQ] �޽��� ����
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
            // ���� ���� ���� �ƴ� ��, ���� ������ ����
            // ------------------------------------------------------------------
            if (V_start_process != previous_V_start_process) {
                if (previous_V_start_process) {
                    viewer->removePointCloud("cloud_pcd");
                    viewer->removeShape("result");
                    viewer->removeShape("angle_line");
                    for (const auto& line_id : volume_line_ids) {
                        viewer->removeShape(line_id);
                    }
                    volume_line_ids.clear();
                }


                // ����Ʈ Ŭ���� ������ �ʱ�ȭ
                resetPointCloudData();

                // ���� ���� ������Ʈ
                previous_V_start_process = V_start_process;

                // Viewer�� ���� ���� �ؽ�Ʈ ������Ʈ
                std::string status_text = "V_start_process: " + std::string(V_start_process ? "True" : "False");
                viewer->removeShape("v_start_process_text");
                viewer->addText(status_text, 20, 350, 20, 1, 1, 1, "v_start_process_text");

                std::cout << "[INFO] V_start_process state changed to "
                    << (V_start_process ? "True" : "False") << ". Data has been reset." << std::endl;
            }


            // ------------------------------------------------------------------
            // 1�� �Ⱦ�

            int count_load_roi_1 = 0;
            float loadROI_x1_min = -fixed_ground_height + 0.1f;
            float loadROI_x1_max = -fixed_ground_height + 2.0f;
            float loadROI_y_min = -0.9f;
            float loadROI_y_max = -0.35f;
            float loadROI_z_min = 0.35f;
            float loadROI_z_max = 0.65f;

            pcl::PointCloud<pcl::PointXYZ>::Ptr Pickup_cloud;

            if (READ_PCD_FROM_FILE) {
                Pickup_cloud = cloud_merge;
            }
            else {
                Pickup_cloud = cloud_raw;
            }

            for (const auto& point : Pickup_cloud->points) {
                if (point.x >= loadROI_x1_min && point.x <= loadROI_x1_max &&
                    point.y >= loadROI_y_min && point.y <= loadROI_y_max &&
                    point.z >= loadROI_z_min && point.z <= loadROI_z_max) {
                    count_load_roi_1++;
                }
            }


            if (count_load_roi_1 >= 1 && count_load_roi_1 <= 1000) {
                PickUp_1 = true;
                //std::cout << "[PICKUP] 1st Floor !!!" << std::endl;
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
                viewer->removeShape("load_roi_box_1");
            }


            // -------------------------------------------------------------------------------------
            // ���� ���� ����
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


                        if (point.x < V_x_max && point.x >= -fixed_ground_height + 0.15f &&
                            point.y >= y_min && point.y <= -0.45f && // �⺻�� -0.45f (�鷹��Ʈ ��)
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

                    // ���� ���� ��� ���
                    if (x_lengths.size() == vector_size) {
                        result_height = calculateAverageX(x_lengths) * 1000;
                        x_lengths.clear();
                    }

                    if (result_height != 0 && result_width == 0 && result_length == 0) {
                        bool result_status = true;

                        float ground_correction_mm = 0.0f;
                        if (ground_height_fixed) {
                            ground_correction_mm = fixed_ground_height * 1000.0f;
                            result_height += (ground_correction_mm + 9.5f);
                        }
                        else {
                            result_height += config.height_threshold + 9.5f;  // ���� �ӽ� ���� (2755)
                        }

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
                        //<< "GroundFix: " << ground_correction_mm << " mm \n"; // �߰�
                        //<< "PCD: " << cloud_pcd_local->points.size() << " cnt ";

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
                // PCD ������ ���� (�ܼ� ����)
                if (SAVE_PCD_FROM_FILE) {
                    *cloud_pcd += *cloud_merge;
                    std::cout << "[DEBUG] Accumulated " << cloud_pcd->size() << " points in cloud_pcd. \n";
                }

                // ���� ����� ���� chunk ��� ���� ó�� ����
                if (READ_PCD_FROM_FILE) {
                    cloud_merge->clear();
                }

                // Ŭ���� ó�� �� Ŭ����_merge �ʱ�ȭ
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
    // ���� �� PCD ����
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
	curl_global_cleanup(); // curl ���� ���� (���α׷� ���� �� �ѹ� ȣ��)
    return 0;
}
