#pragma once
#include <string>
#include "json.hpp"

struct HostInfo {
    std::string lidar_ip;
    std::string config_host_ip;    
};

struct WATAConfig {
	int iteration;

	int mean_k;
	float threshold;
	
	float reach_height;
	float counterbalance_height;

    bool flag_detect_plane_yz = false;
    bool flag_load_roi = false;
    bool flag_raw_cloud = false;
    bool flag_intensity = false;
    bool flag_height_only = false;
    bool flag_reach_off_counter = false;
    bool flag_replay = false;
    bool flag_heart_beat = true;
    bool flag_volume = false;
    bool flag_tuning = false;
	bool flag_tune_all_files = false;
	std::string tune_folder = "";

    bool read_file = false;
    bool save_file = false;
    std::string read_file_name = "";
    std::string save_file_name = "";
};

namespace config {
	HostInfo ReadHostInfo(const std::string& path);
    WATAConfig ReadConfig(const std::string& path);
}