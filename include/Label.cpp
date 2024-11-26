#include "Label.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <filesystem>




BoxParams::BoxParams(int x, int y, int w, int h, int l, bool xywh) :label(l) {
    if (x <= 0. || y <= 0. || w <= 0. || y <= 0.)
        throw std::invalid_argument("Box argument are less than 0.0.");
    if (xywh) {
        x1 = x - w / 2;
        x2 = x + w / 2;
        y1 = y - h / 2;
        y2 = y + h / 2;
    }
    else {
        x1 = std::min(x, w);
        y1 = std::min(y, h);
        x2 = std::max(x, w);
        y2 = std::max(y, h);


    }
 }


void Label::add(const int& x1, const int& y1, const int& x2, const int& y2) {

        boxes.emplace_back(x1, y1, x2, y2, 0, false);
    }

void Label::add_box(BoxParams& box) {
        boxes.emplace_back(box);
    }

void Label::clear() {
        boxes.clear();
    }

int Label::change_label(point p, int num, int num_class) {
        if (boxes.empty()) {
            return -1;
        }
        FP64 distance = DBL_MAX;
        FP64 cur;
        auto nearest_it = boxes.end();
        for (auto it = boxes.begin(); it != boxes.end(); ++it) {
            cur = sqrt((it->x1 - p.x) * (it->x1 - p.x) + (it->y1 - p.y) * (it->y1 - p.y));

            if (cur < distance) {
                nearest_it = it;
                distance = cur;
            }
        }
        int temp_label = nearest_it->label + num;
        if (temp_label < 0) {
            nearest_it->label = num_class - 1;
        }
        else if (temp_label >= num_class)
        {
            nearest_it->label = 0;
        }
        else {
            nearest_it->label = temp_label;
        }
        return  nearest_it->label;
    }

const std::vector<BoxParams>& Label::getBoxes() const {
        return boxes;
    }
int Label::remove_nearest_box(point p) {
        FP64 distance = DBL_MAX;
        FP64 cur;

        if (boxes.empty()) {
            return 0;
        }
        auto nearest_it = boxes.end();
        for (auto it = boxes.begin(); it != boxes.end(); ++it) {
            cur = sqrt((it->x1 - p.x) * (it->x1 - p.x) + (it->y1 - p.y) * (it->y1 - p.y));

            if (cur < distance) {
                nearest_it = it;
                distance = cur;
            }
        }

        if (nearest_it != boxes.end()) {
            boxes.erase(nearest_it);
        }
        return 0;
    }
int Label::to_yolo_file(fs::path& txt_path, cv::Mat& img) {

        fs::path dir_path = txt_path.parent_path();
        if (!fs::exists(dir_path))
            if (!fs::create_directories(dir_path)) {
                std::cerr << "Failed to create directories: " << dir_path << std::endl;
            }


        std::ofstream output_file(txt_path.string());

        // 检查文件是否成功打开
        if (!output_file) {
            std::cerr << "Error opening file for writing!" << std::endl;
            return 1;
        }
        FP64 h = img.size[0];
        FP64 w = img.size[1];
        std::ostringstream oss;
        for (auto& box : boxes) {
            FP64 xc = (box.x1 + box.x2) / 2.;
            FP64 yc = (box.y1 + box.y2) / 2.;
            FP64 wc = std::abs(box.x2 - box.x1);
            FP64 hc = std::abs(box.y2 - box.y1);
            oss << box.label << " " << xc / w << " " << yc / h << " " << wc / w << " " << hc / h << std::endl;
        }

        output_file << oss.str();
        output_file.close();
        return 0;
    }
int Label::read_label_from_txt(fs::path& txt_path, cv::Mat& img) {
        int img_h = img.size[0];
        int img_w = img.size[1];
        FP64 x, y, w, h;
        int label;
        std::string line;
        std::ifstream input_file(txt_path);
        if (!input_file) {
            std::cerr << "Error opening file!" << std::endl;
            return 1;
        }

        while (std::getline(input_file, line)) {
            // 使用 sscanf 或类似方法直接解析数据
            if (sscanf_s(line.c_str(), "%d %lf %lf %lf %lf", &label, &x, &y, &w, &h) == 5) {
                // 计算坐标和尺寸
                int x_int = static_cast<int>(round(x * img_w));
                int y_int = static_cast<int>(round(y * img_h));
                int w_int = static_cast<int>(round(w * img_w));
                int h_int = static_cast<int>(round(h * img_h));

                boxes.emplace_back(x_int, y_int, w_int, h_int, label, true);

            }
        }
        input_file.close();

        return 0;

    }
