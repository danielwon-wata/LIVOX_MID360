#ifndef VTK_UI_HPP
#define VTK_UI_HPP

// vtk_ui.hpp
// Encapsulates all VTK-based UI components for Livox MID360 Visualizer

#include <string>
#include <vector>
#include <atomic>
#include "config.hpp"    // WATAConfig, HostInfo, FovCfg
#include "livox_lidar_api.h" // For FOV callbacks and SDK functions

// main.cpp 등에 정의된 전역 심볼들
#pragma once
extern std::atomic<uint32_t> g_lidar_handle;
extern FovCfg            fov_cfg0;

#include <pcl/visualization/pcl_visualizer.h>
#include <vtkSmartPointer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkTexturedButtonRepresentation2D.h>
#include <vtkButtonWidget.h>
#include <vtkSliderWidget.h>
#include <vtkSliderRepresentation2D.h>
#include <vtkTextActor.h>
#include <vtkPNGReader.h>
#include <vtkCommand.h>
#include <vtkImageData.h>


namespace vtk_ui {
    extern std::atomic<bool> rebootRequested;

    // Forward declarations of callback structs
    struct SliderCallback;
    struct ButtonCallback;
    struct ReachCounterCallback;
    struct ExitCallback;
    struct RebootCallback;
    struct ResetCallback;
    struct FovSliderCallback;
    struct SetFovButtonCallback;

    // Utility loaders
    vtkImageData* LoadPNG(const std::string& path);

    vtkSmartPointer<vtkButtonWidget> MakeButton(
        vtkRenderWindowInteractor* iren,
        vtkRenderer* ui_renderer,
        const std::string& icon_off,
        const std::string& icon_on,
        double bnds[6],
        bool* flag_ptr,
        const std::string& name);

    // Manages all VTK UI elements: buttons, sliders, panels, and callbacks
    class UIManager {
    public:
        UIManager(
            pcl::visualization::PCLVisualizer::Ptr viewer,
            vtkRenderWindowInteractor* interactor,
            WATAConfig& config,
            float& fixed_ground_height,
            bool& heightCalibration_mode,
            bool& onReachoffCounter,
            bool& is_paused,              
            bool& enableIntensity);

        // Build and attach all UI widgets
        void Setup();

        // Called per-frame to refresh dynamic text labels
        void Update();

        void SetConnectionText(const std::string& t);
        void SetDevText(const std::string& t);
        void SetStatusText(const std::string& t);
        void UpdateCameraPositionText(const std::string& t);

    private:
        // Internal builders
        void SetupRenderers();
        void SetupTextPanels();
        void SetupButtons();
        void SetupSliders();
        void SetupCallbacks();

        // References
        pcl::visualization::PCLVisualizer::Ptr viewer_;
        vtkRenderWindowInteractor* interactor_;
        vtkSmartPointer<vtkRenderer> main_renderer_;
        vtkSmartPointer<vtkRenderer> ui_renderer_;

        WATAConfig& config_;
        float& fixed_ground_height_;
        bool& heightCalibration_mode_;
        bool& onReachoffCounter_;
        bool& is_paused_;
        bool& enableIntensity_;

        // Actors and widgets
        vtkSmartPointer<vtkTextActor> status_panel_;
        vtkSmartPointer<vtkTextActor> dev_panel_;
        vtkSmartPointer<vtkTextActor> conn_panel_;
        vtkSmartPointer<vtkTextActor> ground_text_actor_;
		vtkSmartPointer<vtkTextActor> camera_position_text_actor_;

        std::vector<vtkSmartPointer<vtkButtonWidget>> button_widgets_;
        std::vector<vtkSmartPointer<vtkSliderWidget>> slider_widgets_;

        // FOV slider representations
        vtkSmartPointer<vtkSliderRepresentation2D> yaw_start_rep_;
        vtkSmartPointer<vtkSliderRepresentation2D> yaw_stop_rep_;
        vtkSmartPointer<vtkSliderRepresentation2D> pitch_start_rep_;
        vtkSmartPointer<vtkSliderRepresentation2D> pitch_stop_rep_;

        vtkSmartPointer<vtkSliderRepresentation2D> iteration_sliderRep;
		vtkSmartPointer<vtkSliderRepresentation2D> mean_k_sliderRep;
		vtkSmartPointer<vtkSliderRepresentation2D> threshold_sliderRep;
		// Default values for reset
		int default_iter = 1000;
		int default_mean_k = 60;
		float default_threshold = 1.5f;
		FovCfg default_fov0 = { 0, 360, -9, 52 }; // Default FOV configuration
    };

    // --- Callback definitions ---

    // Slider adjusts integer or float fields in WATAConfig
    struct SliderCallback : public vtkCommand {
        static SliderCallback* New();
        WATAConfig* iteration_ptr = nullptr;
        WATAConfig* meank_ptr = nullptr;
        WATAConfig* threshold_ptr = nullptr;
        void Execute(vtkObject* caller, unsigned long, void*) override;
    };

    // Button toggles a boolean flag
    struct ButtonCallback : public vtkCommand {
        static ButtonCallback* New();
        bool* toggleFlag = nullptr;
        std::string name;
        void Execute(vtkObject* caller, unsigned long, void*) override;
    };

    // Combined reach/counter toggle: updates config, ground height text, and button state
    struct ReachCounterCallback : public vtkCommand {
        static ReachCounterCallback* New();
        WATAConfig* config = nullptr;
        bool* toggleFlag = nullptr;
        float* groundHeightPtr = nullptr;
        vtkTextActor* heightTextActor = nullptr;
        vtkButtonWidget* buttonWidget = nullptr;
        void Execute(vtkObject*, unsigned long, void*) override;
    };

    // Exits application cleanly
    struct ExitCallback : public vtkCommand {
        static ExitCallback* New();
        pcl::visualization::PCLVisualizer* viewer = nullptr;
        vtkRenderWindowInteractor* iren = nullptr;
        void Execute(vtkObject*, unsigned long, void*) override;
    };

    // Signals for reboot: closes and re-launches
    struct RebootCallback : public vtkCommand {
        static RebootCallback* New();
        vtkRenderWindowInteractor* iren = nullptr;
        pcl::visualization::PCLVisualizer* viewer = nullptr;
        void Execute(vtkObject*, unsigned long, void*) override;
    };

    // Resets sliders, config values, FOV, and camera position
    struct ResetCallback : public vtkCommand {
        static ResetCallback* New();
        WATAConfig* cfg = nullptr;
        vtkSliderRepresentation2D* iterRep = nullptr;
        vtkSliderRepresentation2D* meanRep = nullptr;
        vtkSliderRepresentation2D* thrRep = nullptr;
        vtkSliderRepresentation2D* yaw0Rep = nullptr;
        vtkSliderRepresentation2D* yaw1Rep = nullptr;
        vtkSliderRepresentation2D* pit0Rep = nullptr;
        vtkSliderRepresentation2D* pit1Rep = nullptr;
        pcl::visualization::PCLVisualizer* viewer = nullptr;
        int default_iter;
        int default_mean_k;
        float default_threshold;
        FovCfg default_fov0;
        void Execute(vtkObject*, unsigned long, void*) override;
    };

    // Slider for FOV parameters: updates FOV live
    struct FovSliderCallback : public vtkCommand {
        static FovSliderCallback* New();
        int field; // 0=yaw_start, 1=yaw_stop, 2=pitch_start, 3=pitch_stop
        vtkTextActor* infoText = nullptr;
        void Execute(vtkObject* caller, unsigned long, void*) override;
    };

    // Applies FOV settings to hardware
    struct SetFovButtonCallback : public vtkCommand {
        static SetFovButtonCallback* New();
        void Execute(vtkObject*, unsigned long, void*) override;
    };

} // namespace vtk_ui

#endif // VTK_UI_HPP
