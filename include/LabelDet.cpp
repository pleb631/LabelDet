﻿#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <cstdio>

#include "LabelDet.h"


    // 全局回调函数，用于处理鼠标事件
void LabelDet::onMouse(int event, int x, int y, int flags, void* userdata) {
        // 将 userdata 转换为 LabelDet* 类型的指针
        LabelDet* self = static_cast<LabelDet*>(userdata);
        mouse_event current_event(x, y, event, flags);
        self->update_event(current_event);

        //std::cout << "Mouse moved to (" << x << ", " << y << ")" << std::endl;

    }


LabelDet::LabelDet(std::string d)  {
        //if (directory.empty())
        //    return;

        if (fs::is_directory(d)) {
            directory = d;
        }

        else if (fs::is_regular_file(d)) {
            directory = fs::path(d).parent_path().string();
        }
        // 初始化，获取目录中的文件

       getFilesInDirectory(directory, files, suffixs);
       cp = fs::path(directory) / "checkpoint";
       if (fs::exists(cp)) {
           read_chekpoint(cp);
       }
       return;
    }
    
int LabelDet::read_chekpoint(fs::path& filename) {
        std::string content_read;
        std::ifstream infile(filename);

        if (infile.is_open()) {
            std::getline(infile, content_read); // 从文件读取内容（按行读取）
            infile.close(); // 关闭文件
        }
        else {
            std::cerr << "无法打开文件进行读取: " << filename << std::endl;
            return 1;
        }
        current_index = std::stoi(content_read);
        return 0;
    }
int LabelDet::write_chekpoint(fs::path& filename) {
        std::string content = std::to_string(current_index);
        std::ofstream outfile(filename); // 打开文件进行写入
        if (outfile.is_open()) {
            outfile << content; // 将内容写入文件
            outfile.close(); // 关闭文件
            return 0;
        }
        else {
            std::cerr << "无法打开文件进行写入: " << filename << std::endl;
            return 1;
        }

    }
void LabelDet::plt_a_box(cv::Mat& image, BoxParams& box,int tl) {

        int h, w;

        
        cv::Scalar color= colors[box.label];
        cv::Point point1, point2;
        h = image.size[0];
        w = image.size[1];
        if (tl == 0)
        {
            tl = std::max(1., round(0.001 * (h + w) / 2));
        }

        cv::rectangle(image, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), color,tl);

        //label
        int tf = std::max(tl - 1, 1);
        double sf = tl / 3.;
        std::string label = std::to_string(box.label);
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, 0,sf,tf,&baseline);

        bool outside = (box.y1 - textSize.width >= 3);
        cv::rectangle(image, cv::Point(box.x1, box.y1), cv::Point(box.x1 + textSize.width, outside ? box.y1 -textSize.height-3 : box.y1 + textSize.height + 3), color, -1, cv::LINE_AA);  // filled
        cv::putText(image, label, cv::Point(box.x1, outside ? box.y1 - 2 : box.y1 + textSize.width + 2), 0, sf, cv::Scalar(255, 255, 255), tf, cv::LINE_AA);
    }
void LabelDet::plot_boxes(cv::Mat& image,const std::vector<BoxParams>& boxes) {


        for (auto box = boxes.begin(); box < boxes.end(); box++) {
            BoxParams box1 = *box;
 
                
            encode_box(box1);
            plt_a_box(image, box1, 2);

        }

    }

int LabelDet::encode_box(BoxParams& box) {
        box.x1 = static_cast<int>(round(box.x1*win_info.r+ win_info.left));
        box.x2 = static_cast<int>(round(box.x2 * win_info.r + win_info.left));
        box.y1 = static_cast<int>(round(box.y1 * win_info.r + win_info.top));
        box.y2 = static_cast<int>(round(box.y2 * win_info.r + win_info.top));

        return 0;
    }


point LabelDet::decode_point(int x,int y) {
        
        return point(static_cast<int>((x - win_info.left) / win_info.r), static_cast<int>((y - win_info.top) / win_info.r));
    }
int LabelDet::decode_box(BoxParams &box) {
        box.x1 = static_cast<int>((box.x1 - win_info.left) / win_info.r);
        box.x2 = static_cast<int>((box.x2 - win_info.left) / win_info.r);
        box.y1 = static_cast<int>((box.y1 - win_info.top) / win_info.r);
        box.y2 = static_cast<int>((box.y2 - win_info.top) / win_info.r);

        return 0;
    }

int LabelDet::update_event(mouse_event& e) {
        
        e.x = e.x + span.p1.x;
        e.y = e.y + span.p1.y;
        event = e;
        int x = e.x;
        int y = e.y;
        if (event.event == cv::EVENT_RBUTTONUP) {
            labels.remove_nearest_box(decode_point(x,y));
        }
        if (event.event == cv::EVENT_LBUTTONDOWN) {
            lastp.x = std::clamp(x, win_info.left, win_info.win_width-win_info.right);
            lastp.y = std::clamp(y,win_info.top, win_info.win_height-win_info.bottom);
        }

        if (event.event == cv::EVENT_LBUTTONUP) {
            int current_x = std::clamp(x, win_info.left, win_info.win_width - win_info.right);
            int current_y = std::clamp(y, win_info.top, win_info.win_height - win_info.bottom);
            if (std::abs(current_x - lastp.x) > 3 && std::abs(current_y - lastp.y) > 3)
            {
                BoxParams box(current_x, current_y,lastp.x,lastp.y,current_clss,false);
                decode_box(box);
                labels.add_box(box);
            }
                
        }
        if (event.event == cv::EVENT_MOUSEWHEEL) {

            if (span.scale == 1) {
                span.p1.x = 0;
                span.p1.y = 0;
                span.p2.x = win_info.win_width;
                span.p2.y = win_info.win_height;
            }

            if (event.flags > 0) {
                span.scale = std::max(span.scale*0.9,0.1);
            }
            else {
                span.scale = std::min(span.scale * 1.1, 1.);
            }

            FP64 new_width_fp = win_info.win_width * span.scale;
            FP64 new_height_fp = win_info.win_height * span.scale;

            int x1 = static_cast<int>(x - (static_cast<FP64>(x - span.p1.x) / (span.p2.x - span.p1.x) * new_width_fp));
            int y1 = static_cast<int>(y - (static_cast<FP64>(y - span.p1.y) / (span.p2.y - span.p1.y) * new_height_fp));
            span.p1.x = std::max(x1, 0);
            span.p1.y = std::max(y1, 0);
            span.p2.x = std::min(static_cast<int>(x1 + new_width_fp), win_info.win_width);
            span.p2.y = std::min(static_cast<int>(y1 + new_height_fp), win_info.win_height);


        }

        return 0;
    }




    // 运行程序
int LabelDet::run() {
        int key;
        bool init = true,reload=true;
        auto last_time = std::chrono::high_resolution_clock::now();
        const int delay_ms = 25;
        bool need_render = false;
        int return_value = -1;


        if (files.empty())
            return 0;

        // 创建窗口
        cv::namedWindow(winName, cv::WINDOW_NORMAL);
        cv::resizeWindow(winName, 640, 480);

        while (1) {
            
            
            if (init || reload) {
                std::cout << "total num: " << files.size() << "  " << "imgID: " << current_index << std::endl << "imgpath: " << files[current_index].string() << std::endl << std::endl;
                img = cv::imread(files[current_index].string());
                fs::path txt_path = imgpath2txtpath(files[current_index]);
                if (fs::exists(txt_path)) {
                    labels.read_label_from_txt(txt_path,img);
                }
                current_img = img.clone();
                if (img.empty())
                    break;
                init = false;
            }

            // 显示图片
            auto current_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time);
            int milliseconds = static_cast<int>(duration.count());
            if (milliseconds < delay_ms)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms - milliseconds));

            }
            last_time = std::chrono::high_resolution_clock::now();
            getScaleInfo();
            key = cv::waitKey(1);
            
            // 设置鼠标回调
            cv::setMouseCallback(winName, onMouse, this);
            render(reload, need_render);
            if (need_render)
                need_render = false;
            if (reload)
                reload = false;
            // 处理按键

            if (key != -1) {
                if (key == 27) // 按下 ESC 键退出
                {
                    process_current_label();
                    break;
                }
                if (key == '\u0044' || key == '\u0064') { // 按下 D 键下一张图片
                    if (current_index < files.size() - 1) {
                        process_current_label();
                        current_index++;
                        write_chekpoint(cp);
                        reload = true;
                    }
                }

                if (key == '\u0061' || key == '\u0041') { // 按下 A 键上一张图片
                    if (current_index > 0) {
                        process_current_label();
                        current_index--;
                        write_chekpoint(cp);
                        reload = true;
                    }
                }
                if (key == '\u0051' || key == '\u0071') { //调整分类
                    return_value = labels.change_label(decode_point(event.x, event.y),-1, num_class);
                    if (return_value >= 0) {
                        current_clss = return_value;
                        need_render = true;
                    }
                }
                if (key == '\u0065' || key == '\u0045') { //调整分类
                    return_value = labels.change_label(decode_point(event.x, event.y), 1,num_class);
                    if (return_value >= 0) {
                        current_clss = return_value;
                        need_render = true;
                    }
                }
            }
        }

        return 1;
    }

void LabelDet::process_current_label() {
        fs::path txt_path = imgpath2txtpath(files[current_index]);
        labels.to_yolo_file(txt_path,img);
        span.reset(win_info.win_width,win_info.win_height);
        labels.clear();
    }

int LabelDet::getScaleInfo() {
        cv::Rect window_rect = cv::getWindowImageRect(winName);
        int interpolation;
        int win_w = window_rect.width;

        if (window_rect.width <= 0 || window_rect.height <= 0) {
            return 0;
        }
        if (window_rect.width == win_info.win_width && window_rect.height == win_info.win_width && win_info.img_h == img.size[0] && win_info.img_w == img.size[1])
            return 0;
        int ori_h = img.size[0];
        int ori_w = img.size[1];
        FP64 r = std::min((FP64)window_rect.width / ori_w, (FP64)window_rect.height / ori_h);

        if (r > 1) {
            r = 1;
        }

        if (r > 1) {
            interpolation = cv::INTER_CUBIC;
        }
        else {
            interpolation = cv::INTER_AREA;
        }

        int new_unpad_w = (int)(round(ori_w * r));
        int new_unpad_h = (int)(round(ori_h * r));
        FP64 dw = (window_rect.width - new_unpad_w)/2.;
        FP64 dh = (window_rect.height - new_unpad_h)/2.;

        int top = (int)round(dh - 0.1);
        int bottom = (int)round(dh + 0.1);
        int left = (int)round(dw - 0.1);
        int right = (int)round(dw + 0.1);

        win_info.top = top;
        win_info.bottom = bottom;
        win_info.left = left;
        win_info.right = right;
        win_info.interpolation = interpolation;
        win_info.win_height = window_rect.height;
        win_info.win_width = window_rect.width;
        win_info.r = r;
        win_info.new_unpad_w = new_unpad_w;
        win_info.new_unpad_h = new_unpad_h;

        return 0;

    }



void LabelDet::render(bool reload,bool need_render) {
        int h,w;
        cv::Scalar color(0, 0, 255), fill(144, 144, 144);
       cv::Mat& showMat= current_img;

        if (event.is_event || need_render)
        {
            cv::Size new_unpad_size(win_info.new_unpad_w, win_info.new_unpad_h);

            img.copyTo(current_img);
            cv::resize(current_img, current_img, new_unpad_size, 0,0,win_info.interpolation);
            cv::copyMakeBorder(current_img, current_img, win_info.top, win_info.bottom, win_info.left, win_info.right, cv::BORDER_CONSTANT, fill);
            h = current_img.size[0];
            w = current_img.size[1];


            if(event.event == cv::EVENT_MOUSEMOVE)
                  if (event.flags == cv::EVENT_FLAG_LBUTTON)
                  {
                      cv::Point current_point(event.x, event.y), last_point(lastp.x, lastp.y);
                      cv::rectangle(current_img, last_point, current_point, colors[current_clss], 2);
                  }
                  else {
                      cv::line(current_img, cv::Point(0, event.y), cv::Point(w, event.y), cv::Scalar(255, 0, 0), 2);
                      cv::line(current_img, cv::Point(event.x, 0), cv::Point(event.x, h), cv::Scalar(255, 0, 0), 2);
                    
                  }
                  
                  
            plot_boxes(current_img, labels.getBoxes());
            if (span.scale < 1) {
                h = current_img.size[0];
                w = current_img.size[1];
                cv::Mat subMat = current_img(cv::Range(std::min(h-1,span.p1.y), std::min(h,span.p2.y)), cv::Range(std::min(w-1,span.p1.x), std::min(w,span.p2.x)));

            showMat = subMat;
            }
            event.is_event = false;
            cv::imshow(winName, showMat);


        }

        if (reload) {
            cv::Size new_unpad_size(win_info.new_unpad_w, win_info.new_unpad_h);
            img.copyTo(current_img);
            cv::resize(current_img, current_img, new_unpad_size, 0, 0, win_info.interpolation);
            cv::copyMakeBorder(current_img, current_img, win_info.top, win_info.bottom, win_info.left, win_info.right, cv::BORDER_CONSTANT, fill);

            plot_boxes(current_img, labels.getBoxes());
            cv::imshow(winName, current_img);
        }

        
    }