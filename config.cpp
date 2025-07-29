#include "config.hpp"
#include <fstream>
#include "json.hpp"
#include <sstream>
#include <regex>
#include <iostream>

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

namespace config {

    HostInfo ReadHostInfo(const std::string& path) {
        HostInfo info;
        info.lidar_ip = "Unknown";
        info.config_host_ip = "Unknown";
        std::ifstream fin(path);

        if (!fin.is_open()) {
            std::cerr << "Could not open host info file: " << path << std::endl;
            return info;
        }

        try {
            json j;
            fin >> j;
            if (j.contains("MID360")
                && j["MID360"].contains("host_net_info")
                && j["MID360"]["host_net_info"].is_array()
                && !j["MID360"]["host_net_info"].empty()
                && j["MID360"]["host_net_info"][0].contains("lidar_ip")
                && j["MID360"]["host_net_info"][0]["lidar_ip"].is_array()
                && !j["MID360"]["host_net_info"][0]["lidar_ip"].empty())
            {
                info.lidar_ip = j["MID360"]["host_net_info"][0]["lidar_ip"][0]
                    .get<std::string>();
            }
            auto& arr = j["MID360"]["host_net_info"];
            if (arr.is_array() && !arr.empty()) {
                info.config_host_ip = arr[0].value("host_ip", "Unknown");
            }
        }
        catch (const std::exception& e) {
            std::cerr << "[Config] JSON parse error in " << path << ": " << e.what() << "\n";
        }

        return info;
    }



	WATAConfig ReadConfig(const std::string& path) {
		WATAConfig cfg;
		std::ifstream f(path);

		if (!f.is_open()) {
			throw std::runtime_error("Could not open config file: " + path);
		}
		std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		f.close();
		
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
            cfg.flag_height_only = f.value("heightOnly", cfg.flag_height_only);
            cfg.flag_reach_off_counter = f.value("reachOffCounter", cfg.flag_reach_off_counter);
            cfg.flag_replay = f.value("replay", cfg.flag_replay);
            cfg.flag_heart_beat = f.value("heartBeat", cfg.flag_heart_beat);
            cfg.flag_volume = f.value("volume", cfg.flag_volume);
            if (f.contains("tuning"))
            {
				const auto& t = f["tuning"];
                cfg.flag_tuning = t.value("enable", cfg.flag_tuning);
				cfg.flag_tune_all_files = t.value("all_files", cfg.flag_tune_all_files);
				cfg.tune_folder = t.value("folder", cfg.tune_folder);
			}
            else {
                cfg.flag_tuning = false;
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

        cfg.reach_height = j.value("reach_height", 0.0f);
        cfg.counterbalance_height = j.value("counterbalance_height", 0.0f);

        return cfg;
    }

}


