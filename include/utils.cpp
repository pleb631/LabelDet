#include "utils.h"

// 获取指定目录及其子目录中的文件
void getFilesInDirectory(const fs::path& directory, std::vector<fs::path>& files, const std::vector<std::string>& extensions) {
    if (fs::exists(directory) && fs::is_directory(directory)) {
        for (const auto& entry : fs::recursive_directory_iterator(directory)) {
            if (fs::is_regular_file(entry)) { // 只获取文件
                if (!extensions.empty())
                {
                    if (std::find(extensions.begin(), extensions.end(), entry.path().extension()) != extensions.end())
                        files.push_back(entry.path());
                }
                else
                    files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());
    }
    else {
        std::cerr << "指定的路径不是一个有效的目录。" << std::endl;
    }
}

fs::path imgpath2txtpath(fs::path& img_path) {
    const std::string p1 = "images", p2 = "labels";
    fs::path save_path;
    int index = 0;
    int last_index = -1;
    for (const auto& part : img_path) {
        if (part.string() == p1) {
            last_index = index;
        }
        ++index;
    }
    index = 0;
    if (last_index != -1) {
        for (const auto& part : img_path) {
            if (index != last_index)
                save_path /= part;
            else {
                save_path /= p2;
            }
            index++;
        }
    }
    else save_path = img_path;
    save_path = save_path.replace_extension(".txt");

    return save_path;

}