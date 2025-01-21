#ifndef LIVOX_HEIGHT_H
#define LIVOX_HEIGHT_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <Eigen/Dense>

// ----------------------------------------------------------------------------
// JSON 구조체
// ----------------------------------------------------------------------------
struct StageROI {
    std::string label;   // 스테이지 이름
    int roi_x_start;     // 스테이지 x 시작 좌표(mm)
    int roi_x_end;       // 스테이지 x 끝 좌표(mm)
};

struct WATAConfig {
    int roi_y_start;     // ROI의 y 시작 좌표(mm)
    int roi_y_end;       // ROI의 y 끝 좌표(mm)
    int roi_z_start;     // ROI의 z 시작 좌표(mm)
    int roi_z_end;       // ROI의 z 끝 좌표(mm)
    int angle;           // 회전 각도(도)
    bool read_file;      // 파일 읽기 여부
    bool save_file;      // 파일 저장 여부
    std::string read_file_name;  // 읽기 파일 이름
    std::string save_file_name;  // 저장 파일 이름
};

struct PalletInfo {
    bool is_pallet;      // 파레트 여부
    float P_height;      // 파레트 높이(mm)
};

// ----------------------------------------------------------------------------
// 클래스 정의
// ----------------------------------------------------------------------------
class LivoxHeightProcessor {
public:
    // 생성자 및 소멸자
    LivoxHeightProcessor();
    ~LivoxHeightProcessor();

    // 주요 처리 함수
    WATAConfig readConfigFromJson(const std::string& filePath);
    PalletInfo identifyPallet(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    void resetPointCloudData();
    void processPoints(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_merge,
        pcl::visualization::PCLVisualizer::Ptr viewer,
        int& previous_index);

    // 시각화 함수
    void visualizeHeight(
        pcl::visualization::PCLVisualizer::Ptr viewer,
        const PalletInfo& pallet_info, int index);
    void removePreviousPalletVisualizations(pcl::visualization::PCLVisualizer::Ptr viewer);

    // 클러스터링
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> performClustering(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered);
    void visualizeClusters(
        pcl::visualization::PCLVisualizer::Ptr viewer,
        const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& P_clusters);

private:
    std::vector<StageROI> stages_;
    std::vector<std::string> cluster_box_ids_;
    std::vector<std::string> pallet_line_ids_;
    std::vector<std::string> pallet_height_text_ids_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_merge_;
    std::mutex mutex_;
};

#endif // LIVOX_HEIGHT_H
