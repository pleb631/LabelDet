#pragma once
#ifndef LABEL_H
#define LABEL_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "utils.h"


struct BoxParams {
    int x1, y1, x2, y2, label = 0;
    BoxParams(int x, int y, int w, int h, int l = 0, bool xywh = true);
    BoxParams() = default;
};

class Label {
private:
    std::vector<BoxParams> boxes;

public:
    void add(const int& x1, const int& y1, const int& x2, const int& y2);
    const std::vector<BoxParams>& getBoxes() const;
    int read_label_from_txt(fs::path& txt_path, cv::Mat& img);
    int to_yolo_file(fs::path& txt_path, cv::Mat& img);
    int remove_nearest_box(point p);
    int change_label(point p, int num, int num_class);
    void clear();
    void add_box(BoxParams& box);


    // 其他方法...
};

#endif // LABEL_H
