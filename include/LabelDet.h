#pragma once
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "Label.h"


struct mouse_event
{
    int x;
    int y;
    int event;
    int flags;
    bool is_event = false;
    mouse_event() :x(0), y(0), event(0), flags(0) {}
    mouse_event(int x, int y, int e, int f) :x(x), y(y), event(e), flags(f) { is_event = true; }
};

struct WinInfo
{
    int left, top, right, bottom, interpolation;
    FP64 r;
    int new_unpad_w;
    int new_unpad_h;
    int img_h = 0, img_w = 0, win_width = 0, win_height = 0;
};

// LabelDet 类
class LabelDet {

private:
    std::vector<fs::path> files;     // 存储文件路径
    const std::string winName = "image";    // 窗口名称
    const std::vector<std::string> suffixs{ ".png", ".jpg" }; // 支持的图片类型
    std::string directory;                // 目录路径
    cv::Mat img;                        // 当前图像
    cv::Mat current_img;
    int current_index = 0;             // 当前图片索引
    int num_class = 4;
    int current_clss = 0;
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0),        // 蓝色        
        cv::Scalar(0, 255, 0),        // 绿色
        cv::Scalar(0, 0, 255),        // 红色
        cv::Scalar(0, 255, 255),      // 黄色
        cv::Scalar(255, 0, 255),      // 紫色（洋红）
        cv::Scalar(255, 255, 0),      // 青色
        cv::Scalar(128, 128, 128),    // 灰色
        cv::Scalar(0, 165, 255),      // 橙色
        cv::Scalar(42, 42, 165),      // 棕色
        cv::Scalar(203, 192, 255),    // 粉色
        cv::Scalar(255, 200, 0),      // 浅蓝色
        cv::Scalar(0, 0, 139),        // 暗红色
        cv::Scalar(0, 0, 0),          // 黑色
    };

    point lastp;
    Label labels;
    WinInfo win_info;
    Span span;
    static void onMouse(int event, int x, int y, int flags, void* userdata);
public:
    mouse_event event{};
    fs::path cp;
    LabelDet(std::string d);
    int read_chekpoint(fs::path& filename);
    int write_chekpoint(fs::path& filename);
    void plt_a_box(cv::Mat& image, BoxParams& box, int tl);
    void plot_boxes(cv::Mat& image, const std::vector<BoxParams>& boxes);
    int encode_box(BoxParams& box);
    point decode_point(int x, int y);
    int decode_box(BoxParams& box);
    int update_event(mouse_event& e);
    int run();
    void process_current_label();
    int getScaleInfo();
    void render(bool reload, bool need_render);
};