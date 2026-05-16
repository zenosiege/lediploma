#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace std::chrono_literals;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class VisualOdometryNode : public rclcpp::Node {
public:
    VisualOdometryNode() : Node("visual_odom_node"), total_theta_(0.0) {

        // Открываем камеру
        cap_.open(2);
        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Не удалось открыть веб-камеру!");
        } else {
            RCLCPP_INFO(this->get_logger(), "Веб-камера успешно открыта.");
            // Ограничиваем разрешение для нормального FPS в виртуалке
            cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        }

        // Настройка окна вывода
        cv::namedWindow("Visual Odom Experiment", cv::WINDOW_NORMAL);
        cv::resizeWindow("Visual Odom Experiment", 640, 480);

        // Инициализация паблишера и OpenCV объектов
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/visual_odom", 10);
        orb_ = cv::ORB::create(2000);
        matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

        // Таймер и переменные для FPS
        last_time_ = std::chrono::steady_clock::now();
        frame_count_ = 0;

        timer_ = this->create_wall_timer(
            33ms, std::bind(&VisualOdometryNode::timer_callback, this));

        RCLCPP_INFO(this->get_logger(), "Узел запущен: Полный режим визуальной одометрии.");
    }

private:
    void timer_callback() {
        if (!cap_.isOpened()) return;

        // --- БЛОК ЗАМЕРА FPS ---
        frame_count_++;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_time_).count();

        if (elapsed >= 1) {
            RCLCPP_INFO(this->get_logger(), "Текущий FPS: %d", frame_count_);
            frame_count_ = 0;
            last_time_ = now;
        }
        // -----------------------

        cv::Mat current_frame;
        cap_ >> current_frame;

        if (current_frame.empty()) {
            RCLCPP_WARN(this->get_logger(), "Получен пустой кадр с камеры!");
            return;
        }

        try {
            // --- БЛОК ВИЗУАЛЬНОЙ ОДОМЕТРИИ ---
            std::vector<cv::KeyPoint> curr_kp;
            cv::Mat curr_desc;

            orb_->detectAndCompute(current_frame, cv::noArray(), curr_kp, curr_desc);
            if (curr_desc.empty()) return;

            if (!prev_desc_.empty()) {
                std::vector<cv::DMatch> matches;
                matcher_->match(prev_desc_, curr_desc, matches);
                std::sort(matches.begin(), matches.end());

                std::vector<cv::Point2f> src_pts, dst_pts;
                std::vector<cv::DMatch> good_matches;

                for (size_t i = 0; i < std::min(matches.size(), (size_t)2000); ++i) {
                    if (matches[i].distance < 50.0) {
                        src_pts.push_back(prev_kp_[matches[i].queryIdx].pt);
                        dst_pts.push_back(curr_kp[matches[i].trainIdx].pt);
                        good_matches.push_back(matches[i]);
                    }
                }

                if (src_pts.size() > 15) {
                    // Фокусное расстояние "от балды", для реальной точности нужна калибровка
                    double focal_length = 547.4;
                    cv::Mat mask;
                    cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 3.0, mask);

                    if (!H.empty()) {
                        double sum_dtheta = 0;
                        int inliers_cnt = 0;
                        double cx = current_frame.cols / 2.0;

                        for (int i = 0; i < mask.rows; i++) {
                            if (mask.at<uchar>(i)) {
                                double theta_prev = atan2(src_pts[i].x - cx, focal_length);
                                double theta_curr = atan2(dst_pts[i].x - cx, focal_length);
                                sum_dtheta += (theta_curr - theta_prev);
                                inliers_cnt++;
                            }
                        }

                        if (inliers_cnt > 10) {
                            double delta_theta = (sum_dtheta / inliers_cnt);

                            if (std::abs(delta_theta) < 0.001) {
                                delta_theta = 0.0;
                            }

                            if (std::abs(delta_theta) < 0.5) {
                                total_theta_ += delta_theta;
                                total_theta_ = atan2(sin(total_theta_), cos(total_theta_));

                                // Публикуем одометрию в топик
                                publish_odom(this->now());

                                double degrees = total_theta_ * 180.0 / M_PI;
                                // Раскомментируй, если хочешь видеть угол в консоли вместе с FPS
                                RCLCPP_INFO(this->get_logger(), "Угол: %.2f град. | Inliers: %d", degrees, inliers_cnt);
                            }
                        }
                    }

                    // Отрисовка совпадений фичей (рисует линии между старым и новым кадром)
                    cv::Mat img_matches;
                    std::vector<cv::DMatch> final_matches;
                    if (!mask.empty()) {
                        for (int i = 0; i < mask.rows; ++i) {
                            if (mask.at<uchar>(i)) final_matches.push_back(good_matches[i]);
                        }
                    }
                    cv::drawMatches(prev_frame_, prev_kp_, current_frame, curr_kp, final_matches, img_matches);

                    // Показываем окно с совпадениями
                    cv::imshow("Visual Odom Experiment", img_matches);
                    cv::waitKey(1);
                }
            }

            prev_frame_ = current_frame.clone();
            prev_kp_ = curr_kp;
            prev_desc_ = curr_desc.clone();

        } catch (cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "OpenCV error: %s", e.what());
        }
    }

    void publish_odom(const rclcpp::Time & stamp) {
        auto odom_msg = nav_msgs::msg::Odometry();
        odom_msg.header.stamp = stamp;
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = "base_link";

        odom_msg.pose.pose.orientation.z = std::sin(total_theta_ / 2.0);
        odom_msg.pose.pose.orientation.w = std::cos(total_theta_ / 2.0);

        odom_pub_->publish(odom_msg);
    }

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture cap_;

    std::chrono::steady_clock::time_point last_time_;
    int frame_count_;

    cv::Ptr<cv::ORB> orb_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    cv::Mat prev_frame_, prev_desc_;
    std::vector<cv::KeyPoint> prev_kp_;
    double total_theta_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VisualOdometryNode>());
    rclcpp::shutdown();
    return 0;
}
