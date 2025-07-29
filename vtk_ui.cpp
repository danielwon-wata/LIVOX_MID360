#include "vtk_ui.hpp"
#include <atomic>
#include "config.hpp"  // WATAConfig, FovCfg

#include <map>
#include <sstream>
#include <vector>
#include <vtkPNGReader.h>
#include <vtkTexturedButtonRepresentation2D.h>
#include <vtkRendererCollection.h>
#include <vtkTextProperty.h>
#include <vtkImageSliceMapper.h>
#include <vtkImageSlice.h>
#include <vtkRenderWindow.h>
#include <vtkNew.h>
#include <vtkCommand.h>
#include <curl/curl.h>
#include "livox_lidar_api.h"  // for FOV callbacks

// Global image cache
static std::map<std::string, vtkImageData*> imageCache;

namespace vtk_ui {

    //------------------------------------------------------------------------------
    // Utility implementations
    //------------------------------------------------------------------------------

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
        vtkRenderer* ui_renderer,
        const std::string& icon_off,
        const std::string& icon_on,
        double bnds[6],
        bool* flag_ptr,
        const std::string& name)
    {
        auto rep = vtkSmartPointer<vtkTexturedButtonRepresentation2D>::New();
        rep->SetNumberOfStates(2);
        rep->SetButtonTexture(0, LoadPNG(icon_off));
        rep->SetButtonTexture(1, LoadPNG(icon_on));
        rep->SetRenderer(ui_renderer);
        rep->PlaceWidget(bnds);
        if (flag_ptr) rep->SetState(*flag_ptr ? 1 : 0);

        auto btn = vtkSmartPointer<vtkButtonWidget>::New();
        btn->SetInteractor(iren);
        btn->SetCurrentRenderer(ui_renderer);
        btn->SetRepresentation(rep);
        btn->On();
        if (flag_ptr) {
            auto cb = vtkSmartPointer<ButtonCallback>::New();
            cb->toggleFlag = flag_ptr;
            cb->name = name;
            btn->AddObserver(vtkCommand::StateChangedEvent, cb);
        }
        return btn;
    }

    //------------------------------------------------------------------------------
    // Callback "New" factories
    //------------------------------------------------------------------------------
    SliderCallback* SliderCallback::New() { return new SliderCallback(); }
    ButtonCallback* ButtonCallback::New() { return new ButtonCallback(); }
    ReachCounterCallback* ReachCounterCallback::New() { return new ReachCounterCallback(); }
    ExitCallback* ExitCallback::New() { return new ExitCallback(); }
    RebootCallback* RebootCallback::New() { return new RebootCallback(); }
    ResetCallback* ResetCallback::New() { return new ResetCallback(); }
    FovSliderCallback* FovSliderCallback::New() { return new FovSliderCallback(); }
    SetFovButtonCallback* SetFovButtonCallback::New() { return new SetFovButtonCallback(); }

    //------------------------------------------------------------------------------
    // Callback implementations
    //------------------------------------------------------------------------------

    void SliderCallback::Execute(vtkObject* caller, unsigned long, void*) {
        auto slider = static_cast<vtkSliderWidget*>(caller);
        double v = static_cast<vtkSliderRepresentation2D*>(slider->GetRepresentation())->GetValue();
        if (iteration_ptr) iteration_ptr->iteration = static_cast<int>(v);
        if (meank_ptr)     meank_ptr->mean_k = static_cast<int>(v);
        if (threshold_ptr) threshold_ptr->threshold = static_cast<float>(v);
    }

    void ButtonCallback::Execute(vtkObject*, unsigned long, void*) {
        if (toggleFlag) *toggleFlag = !*toggleFlag;
    }

    void ReachCounterCallback::Execute(vtkObject*, unsigned long, void*) {
        config->flag_reach_off_counter = !config->flag_reach_off_counter;
        float h = config->flag_reach_off_counter ? (config->reach_height / 1000.0f)
            : (config->counterbalance_height / 1000.0f);
        *groundHeightPtr = h;
        auto rep = static_cast<vtkTexturedButtonRepresentation2D*>(buttonWidget->GetRepresentation());
        rep->SetState(config->flag_reach_off_counter ? 1 : 0);
        std::ostringstream ss;
        ss << "Current Ground Height: " << h << " m";
        heightTextActor->SetInput(ss.str().c_str());
    }

    void ExitCallback::Execute(vtkObject*, unsigned long, void*) {
        if (iren)   iren->TerminateApp();
        if (viewer) viewer->close();
        LivoxLidarSdkUninit();
        curl_global_cleanup();
        std::exit(0);
    }

    void RebootCallback::Execute(vtkObject*, unsigned long, void*) {
        if (iren)   iren->TerminateApp();
        if (viewer) viewer->close();
        vtk_ui::rebootRequested.store(true);
    }

    void ResetCallback::Execute(vtkObject*, unsigned long, void*) {
        cfg->iteration = default_iter;
        cfg->mean_k = default_mean_k;
        cfg->threshold = default_threshold;
        iterRep->SetValue(default_iter);
        meanRep->SetValue(default_mean_k);
        thrRep->SetValue(default_threshold);
        yaw0Rep->SetValue(default_fov0.yaw_start);
        yaw1Rep->SetValue(default_fov0.yaw_stop);
        pit0Rep->SetValue(default_fov0.pitch_start);
        pit1Rep->SetValue(default_fov0.pitch_stop);

        if (viewer) {
            auto h = g_lidar_handle.load();
            if (h) {
                SetLivoxLidarFovCfg0(h, &default_fov0, nullptr, nullptr);
                EnableLivoxLidarFov(h, 1, nullptr, nullptr);
            }
            viewer->setCameraPosition(4.14367, 5.29453, -3.91817,
                0.946026, -0.261667, 0.191218);
        }
    }

    void FovSliderCallback::Execute(vtkObject* caller, unsigned long, void*) {
        auto slider = static_cast<vtkSliderWidget*>(caller);
        double v = static_cast<vtkSliderRepresentation2D*>(slider->GetRepresentation())->GetValue();
        switch (field) {
        case 0: fov_cfg0.yaw_start = v; break;
        case 1: fov_cfg0.yaw_stop = v; break;
        case 2: fov_cfg0.pitch_start = v; break;
        case 3: fov_cfg0.pitch_stop = v; break;
        }
        if (auto h = g_lidar_handle.load()) {
            SetLivoxLidarFovCfg0(h, &fov_cfg0, nullptr, nullptr);
            EnableLivoxLidarFov(h, 1, nullptr, nullptr);
        }
        if (infoText) {
            std::ostringstream ss;
            ss << "FOV yaw=[" << fov_cfg0.yaw_start << "," << fov_cfg0.yaw_stop << "]"
                << " pitch=[" << fov_cfg0.pitch_start << "," << fov_cfg0.pitch_stop << "]";
            infoText->SetInput(ss.str().c_str());
        }
    }

    void SetFovButtonCallback::Execute(vtkObject*, unsigned long, void*) {
        if (auto h = g_lidar_handle.load()) {
            SetLivoxLidarFovCfg0(h, &fov_cfg0, nullptr, nullptr);
            EnableLivoxLidarFov(h, 1, nullptr, nullptr);
        }
    }

    //------------------------------------------------------------------------------
    // UIManager implementation
    //------------------------------------------------------------------------------

    UIManager::UIManager(
        pcl::visualization::PCLVisualizer::Ptr viewer,
        vtkRenderWindowInteractor* interactor,
        WATAConfig& config,
        float& fixed_ground_height,
        bool& heightCalibration_mode,
        bool& onReachoffCounter,
		bool& is_paused,
        bool& enableIntensity)
        : viewer_(viewer)
        , interactor_(interactor)
        , config_(config)
        , fixed_ground_height_(fixed_ground_height)
        , heightCalibration_mode_(heightCalibration_mode)
        , onReachoffCounter_(onReachoffCounter)
        , is_paused_(is_paused)
        , enableIntensity_(enableIntensity)
    {
    }

    void UIManager::Setup() {
        SetupRenderers();
        SetupTextPanels();
        SetupButtons();
        SetupSliders();
        SetupCallbacks();
    }

    void UIManager::Update() {
        std::ostringstream ss;
        ss << "Ground Height: " << fixed_ground_height_ << " m";
        ground_text_actor_->SetInput(ss.str().c_str());

		UpdateCameraPositionText("");
    }

    void UIManager::SetupRenderers() {

        // 1) RenderWindow 세팅
        auto rw = viewer_->getRenderWindow();
        rw->SetSize(1280, 720);
        rw->SetAlphaBitPlanes(1);
        rw->SetMultiSamples(0);
        rw->SetNumberOfLayers(3);

        // 2) 좌측(0.0~0.75) 메인 3D 렌더러
        auto mainRen = viewer_->getRendererCollection()->GetFirstRenderer();
        mainRen->SetViewport(0.0, 0.0, 0.75, 1.0);
        mainRen->SetLayer(0);
        main_renderer_ = mainRen;      // ← 저장

        // 3) 우측(0.75~1.0) UI 전용 렌더러
        ui_renderer_ = vtkSmartPointer<vtkRenderer>::New();
        ui_renderer_->SetViewport(0.75, 0.0, 1.0, 1.0);
        ui_renderer_->InteractiveOff();
        ui_renderer_->SetLayer(1);
        rw->AddRenderer(ui_renderer_);

        // 4) 로고용 렌더러 (레이어 2)
        auto logoRen = vtkSmartPointer<vtkRenderer>::New();
        logoRen->SetLayer(2);
        logoRen->InteractiveOff();
        logoRen->SetViewport(0.00, 0.00, 0.10, 0.10);
        logoRen->SetBackgroundAlpha(0.0);
        rw->AddRenderer(logoRen);

        // PNG 리더로 로고 이미지 로드
        vtkNew<vtkPNGReader> logoReader;
        logoReader->SetFileName("icons/logo.png");
        logoReader->Update();

        vtkNew<vtkImageSliceMapper> sliceMapper;
        sliceMapper->SetInputConnection(logoReader->GetOutputPort());

        vtkNew<vtkImageSlice> slice;
        slice->SetMapper(sliceMapper);
        logoRen->AddViewProp(slice);

        // UI 렌더러에서 컬러·깊이 버퍼 유지 해제
        ui_renderer_->SetPreserveColorBuffer(false);
        ui_renderer_->SetPreserveDepthBuffer(false);
    }

    void UIManager::SetupTextPanels() {
        dev_panel_ = vtkSmartPointer<vtkTextActor>::New();
        dev_panel_->GetTextProperty()->SetFontSize(18);
        dev_panel_->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
        dev_panel_->GetTextProperty()->SetBackgroundColor(0, 0, 0);
        dev_panel_->GetTextProperty()->SetBackgroundOpacity(0.6);
        dev_panel_->GetPositionCoordinate()->SetCoordinateSystemToNormalizedDisplay();
        dev_panel_->GetPositionCoordinate()->SetValue(0.5, 0.10);
        main_renderer_->AddActor2D(dev_panel_);

        conn_panel_ = vtkSmartPointer<vtkTextActor>::New();
        conn_panel_->GetTextProperty()->SetFontSize(14);
        conn_panel_->GetTextProperty()->SetColor(1.0, 1.0, 0.0);
        conn_panel_->GetTextProperty()->SetBackgroundColor(0, 0, 0);
        conn_panel_->GetTextProperty()->SetBackgroundOpacity(0.6);
        conn_panel_->GetPositionCoordinate()->SetCoordinateSystemToNormalizedDisplay();
        conn_panel_->GetPositionCoordinate()->SetValue(0.01, 0.875);
        main_renderer_->AddActor2D(conn_panel_);

        ground_text_actor_ = vtkSmartPointer<vtkTextActor>::New();
        ground_text_actor_->GetTextProperty()->SetFontSize(16);
        ground_text_actor_->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
        ground_text_actor_->GetPositionCoordinate()->SetCoordinateSystemToNormalizedDisplay();
        ground_text_actor_->GetPositionCoordinate()->SetValue(0.76, 0.917);
        ui_renderer_->AddActor2D(ground_text_actor_);

        status_panel_ = vtkSmartPointer<vtkTextActor>::New();
        status_panel_->GetTextProperty()->SetFontSize(16);
        status_panel_->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
        status_panel_->GetTextProperty()->SetBackgroundColor(0.0, 0.0, 0.0);
        status_panel_->GetTextProperty()->SetBackgroundOpacity(0.6);
		status_panel_->GetPositionCoordinate()->SetCoordinateSystemToNormalizedDisplay();
		status_panel_->GetPositionCoordinate()->SetValue(0.30, 0.90);
        main_renderer_->AddActor2D(status_panel_);

		camera_position_text_actor_ = vtkSmartPointer<vtkTextActor>::New();
		camera_position_text_actor_->GetTextProperty()->SetFontSize(12);
		camera_position_text_actor_->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
		camera_position_text_actor_->GetPositionCoordinate()->SetCoordinateSystemToNormalizedDisplay();
		camera_position_text_actor_->GetPositionCoordinate()->SetValue(0.01, 0.98);
		main_renderer_->AddActor2D(camera_position_text_actor_);

    }

    void UIManager::SetConnectionText(const std::string& t) {
        conn_panel_->SetInput(t.c_str());
    }

    void UIManager::SetDevText(const std::string& t) {
        dev_panel_->SetInput(t.c_str());
    }

    void UIManager::SetStatusText(const std::string& t) {
        status_panel_->SetInput(t.c_str());
    }

    void UIManager::UpdateCameraPositionText(const std::string& t) {
        pcl::visualization::Camera cam;
        viewer_->getCameraParameters(cam);

        std::ostringstream ss;
        ss << "Camera Pos: ("
            << cam.pos[0] << "," << cam.pos[1] << "," << cam.pos[2] << ", "
            << cam.pos[3] << "," << cam.pos[4] << "," << cam.pos[5] << ")";
        camera_position_text_actor_->SetInput(ss.str().c_str());
    }

    void UIManager::SetupButtons() {
        struct Btn {
            std::string off, on;
            double      x, y;
            bool* flag;
            std::string name;

            Btn(const std::string& off_,
                const std::string& on_,
                double x_,
                double y_,
                bool* flag_,
                const std::string& name_)
                : off(off_)
                , on(on_)
                , x(x_)
                , y(y_)
                , flag(flag_)
                , name(name_)
            {}
        };
        std::vector<Btn> btns;
        btns.reserve(10);
        btns.emplace_back("icons/icon_yz_off_black_filled.png", "icons/icon_yz_on_black_filled.png", 0.00, 0.95, &config_.flag_detect_plane_yz, "YZ");
		btns.emplace_back("icons/icon_roi_off_black_filled.png", "icons/icon_roi_on_black_filled.png", 0.05, 0.95, &config_.flag_load_roi, "ROI");
		btns.emplace_back("icons/icon_raw_off_black_filled (2).png", "icons/icon_raw_on_black_filled.png", 0.10, 0.95, &config_.flag_raw_cloud, "RAW");
		btns.emplace_back("icons/icon_intensity_off_reflection.png", "icons/icon_intensity_on_reflection.png", 0.15, 0.95, &enableIntensity_, "Intensity");
		btns.emplace_back("icons/icon_height_width_length.png", "icons/icon_height_only.png", 0.00, 0.861, &config_.flag_height_only, "HeightOnly");
		btns.emplace_back("icons/icon_forklift_counter.png", "icons/icon_forklift_reach.png", 0.15, 0.861, &onReachoffCounter_, "Reach <-> CounterBalace");
		btns.emplace_back("icons/icon_play.png", "icons/icon_stop.png", 0.05, 0.861, &is_paused_, "PlayStop");
		btns.emplace_back("icons/icon_loop_off_black_filled.png", "icons/icon_loop_on_black_filled.png", 0.10, 0.861, &config_.flag_replay, "Loop");
		btns.emplace_back("icons/icon_heartbeat_off_black_filled.png", "icons/icon_heartbeat_on_black_filled.png", 0.00, 0.772, &config_.flag_heart_beat, "HB");
		btns.emplace_back("icons/icon_volume_off.png", "icons/icon_volume_on.png", 0.05, 0.772, &config_.flag_volume, "Volume");
		btns.emplace_back("icons/icon_tuning_off.png", "icons/icon_tuning_on.png", 0.10, 0.772, &config_.flag_tuning, "Tuning");


        int reachBtnIndex = 0;
        for (int i = 0; i < btns.size(); ++i) {
            if (btns[i].name == "Reach <-> CounterBalace") {
                reachBtnIndex = i;
                break;
            }
        }
        auto size = viewer_->getRenderWindow()->GetSize();
        for (auto& b : btns) {
            double x0 = size[0] * b.x, x1 = x0 + 128, y1 = size[1] * b.y, y0 = y1 - 128, bnds[] = { x0,x1,y0,y1,0,0 };
            button_widgets_.push_back(MakeButton(interactor_, ui_renderer_, b.off, b.on, bnds, b.flag, b.name));
        }
        {
            auto reachWidget = button_widgets_[reachBtnIndex];
            auto rcCb = vtkSmartPointer<ReachCounterCallback>::New();
            rcCb->config = const_cast<WATAConfig*>(&config_);
            rcCb->groundHeightPtr = &fixed_ground_height_;
            rcCb->heightTextActor = ground_text_actor_.Get();
            rcCb->buttonWidget = reachWidget.Get();
            reachWidget->AddObserver(vtkCommand::StateChangedEvent, rcCb);
        }

        double exB[6] = { size[0] * 0.20, size[0] * 0.20 + 128, size[1] * 1.05 - 64, size[1] * 1.05,0,0 };
        auto exitBtn = MakeButton(interactor_, ui_renderer_,
            "icons/exit.png", "", exB, nullptr, "Exit");
        auto cbE = vtkSmartPointer<ExitCallback>::New();
        cbE->iren = interactor_; cbE->viewer = viewer_.get();
        exitBtn->AddObserver(vtkCommand::StateChangedEvent, cbE);
        button_widgets_.push_back(exitBtn);

        // Reboot 버튼
        double rbB[6] = { size[0] * 0.124, size[0] * 0.124 + 128, size[1] * 1.05 - 64, size[1] * 1.05,0,0 };
        auto rebootBtn = MakeButton(interactor_, ui_renderer_,
            "icons/reboot.png", "", rbB, nullptr, "Reboot");
        auto cbR = vtkSmartPointer<RebootCallback>::New();
        cbR->iren = interactor_; cbR->viewer = viewer_.get();
        rebootBtn->AddObserver(vtkCommand::StateChangedEvent, cbR);
        button_widgets_.push_back(rebootBtn);
    }


    void UIManager::SetupSliders() {
        iteration_sliderRep = vtkSmartPointer<vtkSliderRepresentation2D>::New();
        mean_k_sliderRep = vtkSmartPointer<vtkSliderRepresentation2D>::New();
        threshold_sliderRep = vtkSmartPointer<vtkSliderRepresentation2D>::New();

        iteration_sliderRep->SetMinimumValue(250);
        iteration_sliderRep->SetMaximumValue(1000);
        iteration_sliderRep->SetValue(config_.iteration);
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
        iteration_sliderRep->SetRenderer(ui_renderer_);

        vtkSmartPointer<vtkSliderWidget> iteration_sliderWidget =
            vtkSmartPointer<vtkSliderWidget>::New();
        iteration_sliderWidget->SetInteractor(interactor_);
        iteration_sliderWidget->SetCurrentRenderer(ui_renderer_);
        iteration_sliderWidget->SetRepresentation(iteration_sliderRep);
        iteration_sliderWidget->SetAnimationModeToAnimate();
        iteration_sliderWidget->KeyPressActivationOff();
        iteration_sliderWidget->SetEnabled(1);

        //iteration_sliderWidget->On();
        auto iteration_scb = vtkSmartPointer<SliderCallback>::New();
        iteration_scb->iteration_ptr = const_cast<WATAConfig*>(&config_);
        iteration_sliderWidget->AddObserver(vtkCommand::InteractionEvent, iteration_scb);

		slider_widgets_.push_back(iteration_sliderWidget);

        mean_k_sliderRep->SetMinimumValue(1);
        mean_k_sliderRep->SetMaximumValue(100);
        mean_k_sliderRep->SetValue(config_.mean_k);
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
        mean_k_sliderRep->SetRenderer(ui_renderer_);

        vtkSmartPointer<vtkSliderWidget> mean_k_sliderWidget =
            vtkSmartPointer<vtkSliderWidget>::New();
        mean_k_sliderWidget->SetInteractor(interactor_);
        mean_k_sliderWidget->SetCurrentRenderer(ui_renderer_);
        mean_k_sliderWidget->SetRepresentation(mean_k_sliderRep);
        mean_k_sliderWidget->SetAnimationModeToAnimate();
        mean_k_sliderWidget->KeyPressActivationOff();
        mean_k_sliderWidget->SetEnabled(1);

        //sliderWidget->On();
        vtkSmartPointer<SliderCallback> scb = vtkSmartPointer<SliderCallback>::New();
        scb->meank_ptr = const_cast<WATAConfig*>(&config_);
        mean_k_sliderWidget->AddObserver(vtkCommand::InteractionEvent, scb);

		slider_widgets_.push_back(mean_k_sliderWidget);

        threshold_sliderRep->SetMinimumValue(0.5);
        threshold_sliderRep->SetMaximumValue(2.5);
        threshold_sliderRep->SetValue(config_.threshold);
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
        threshold_sliderRep->SetRenderer(ui_renderer_);
        vtkSmartPointer<vtkSliderWidget> threshold_sliderWidget =
            vtkSmartPointer<vtkSliderWidget>::New();
        threshold_sliderWidget->SetInteractor(interactor_);
        threshold_sliderWidget->SetCurrentRenderer(ui_renderer_);
        threshold_sliderWidget->SetRepresentation(threshold_sliderRep);
        threshold_sliderWidget->SetAnimationModeToAnimate();
        threshold_sliderWidget->KeyPressActivationOff();
        threshold_sliderWidget->SetEnabled(1);

        //threshold_sliderWidget->On();
        vtkSmartPointer<SliderCallback> threshold_scb = vtkSmartPointer<SliderCallback>::New();
        threshold_scb->threshold_ptr = const_cast<WATAConfig*>(&config_);
        threshold_sliderWidget->AddObserver(vtkCommand::InteractionEvent, threshold_scb);

		slider_widgets_.push_back(threshold_sliderWidget);

        vtkSmartPointer<vtkTextActor> fovInfoText = vtkSmartPointer<vtkTextActor>::New();
        fovInfoText->GetTextProperty()->SetFontSize(16);
        fovInfoText->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
        fovInfoText->SetPosition(20, 10);           // 적당한 위치
        ui_renderer_->AddActor2D(fovInfoText);
        fovInfoText->SetInput("FOV yaw = [0,360]\npitch = [-9,52]");


        // 수평 시작
        auto makeFovSlider = [&](double minV, double maxV, double yNorm,
            const char* title, int field,
            double initialValue,
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
            rep->SetRenderer(ui_renderer_);

            auto widget = vtkSmartPointer<vtkSliderWidget>::New();
            widget->SetInteractor(interactor_);
            widget->SetCurrentRenderer(ui_renderer_);
            widget->SetRepresentation(rep);
            widget->SetAnimationModeToAnimate();
            widget->KeyPressActivationOff();
            widget->SetEnabled(1);
            widget->On();



            auto cb = vtkSmartPointer<FovSliderCallback>::New();
            cb->field = field;
            cb->infoText = fovInfoText.Get();
            widget->AddObserver(vtkCommand::InteractionEvent, cb);

            slider_widgets_.push_back(widget);

            outRep = rep;
        };

        // 네 개의 슬라이더 달기
        makeFovSlider(0, 360, 0.25, "Yaw Start", 0, fov_cfg0.yaw_start, yaw_start_rep_);
        makeFovSlider(0, 360, 0.20, "Yaw Stop", 1, fov_cfg0.yaw_stop, yaw_stop_rep_);
        makeFovSlider(-9, 52, 0.15, "Pitch Start", 2, fov_cfg0.pitch_start, pitch_start_rep_);
        makeFovSlider(-9, 52, 0.10, "Pitch Stop", 3, fov_cfg0.pitch_stop, pitch_stop_rep_);
    }

    void UIManager::SetupCallbacks() {
		int* winSize = viewer_->getRenderWindow()->GetSize();
        auto resetRep = vtkSmartPointer<vtkTexturedButtonRepresentation2D>::New();
        resetRep->SetNumberOfStates(1);
        resetRep->SetButtonTexture(0, LoadPNG("icons/reset.png"));
        double rx0 = winSize[0] * 0.075, rx1 = rx0 + 128;
        double ry1 = winSize[1] * 1.05, ry0 = ry1 - 64;
        double rb[6] = { rx0,rx1, ry0,ry1, 0,0 };
        resetRep->PlaceWidget(rb);
        resetRep->SetRenderer(ui_renderer_);

        auto resetBtn = vtkSmartPointer<vtkButtonWidget>::New();
        resetBtn->SetInteractor(interactor_);
        resetBtn->SetCurrentRenderer(ui_renderer_);
        resetBtn->SetRepresentation(resetRep);
        resetBtn->On();

        // 콜백 연결
        auto cbReset = vtkSmartPointer<ResetCallback>::New();
        cbReset->cfg = const_cast<WATAConfig*>(&config_);
        cbReset->iterRep = iteration_sliderRep;
        cbReset->meanRep = mean_k_sliderRep;
        cbReset->thrRep = threshold_sliderRep;
        cbReset->yaw0Rep = yaw_start_rep_;
        cbReset->yaw1Rep = yaw_stop_rep_;
        cbReset->pit0Rep = pitch_start_rep_;
        cbReset->pit1Rep = pitch_stop_rep_;
        cbReset->viewer = viewer_.get();
        cbReset->default_iter = default_iter;
        cbReset->default_mean_k = default_mean_k;
        cbReset->default_threshold = default_threshold;
        cbReset->default_fov0 = default_fov0;
        resetBtn->AddObserver(vtkCommand::StateChangedEvent, cbReset);

		button_widgets_.push_back(resetBtn);
    }

} // namespace vtk_ui
