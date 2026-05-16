#include <rclcpp/rclcpp.hpp>
// #include <nav_msgs/msg/odometry.hpp>
#include <opencv2/opencv.hpp>
// #include <opencv2/features2d.hpp>
// #include <opencv2/calib3d.hpp>
#include <vector>
// #include <cmath>
// #include <algorithm>
#include <chrono>

using namespace std::chrono_literals;

// #ifndef M_PI
// #define M_PI 3.14159265358979323846
// #endif

class VisualOdometryNode : public rclcpp::Node {
public:
    VisualOdometryNode() : Node("visual_odom_node") {
        // Открываем через V4L2
        cap_.open(2, cv::CAP_V4L2);

        if (cap_.isOpened()) {
            // Жестко ограничиваем буфер и разрешение
            cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
            cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

            // Опционально: можно попробовать задать формат MJPG, он легче для USB
            cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));

            RCLCPP_INFO(this->get_logger(), "Камера настроена на скоростной режим.");
        }

        cv::namedWindow("Raw Camera Frame", cv::WINDOW_NORMAL);
        cv::resizeWindow("Raw Camera Frame", 640, 480);

        timer_ = this->create_wall_timer(
            30ms, std::bind(&VisualOdometryNode::timer_callback, this));
    }

private:
    void timer_callback() {
        if (!cap_.isOpened()) return;

        cv::Mat current_frame;
        cap_ >> current_frame;

        if (current_frame.empty()) {
            RCLCPP_WARN(this->get_logger(), "Получен пустой кадр с камеры!");
            return;
        }

        try {
            /*
            ================================================================
               БЛОК ВИЗУАЛЬНОЙ ОДОМЕТРИИ ВРЕМЕННО ЗАКОММЕНТИРОВАН
            ================================================================
            std::vector<cv::KeyPoint> curr_kp;
            cv::Mat curr_desc;

            orb_->detectAndCompute(current_frame, cv::noArray(), curr_kp, curr_desc);
            if (curr_desc.empty()) return;

            if (!prev_desc_.empty()) {
                ... весь код с RANSAC, вычислением углов и публикацией ...
            }

            prev_frame_ = current_frame.clone();
            prev_kp_ = curr_kp;
            prev_desc_ = curr_desc.clone();
            ================================================================
            */

            // Две строчки для вывода изображения, как ты просил:
            cv::imshow("Raw Camera Frame", current_frame);         // Показываем чистый кадр для проверки
            // cv::imshow("Visual Odom Experiment", img_matches);  // Отрисовка фичей одометрии (пока отключено)

            cv::waitKey(1);

        } catch (cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "OpenCV error: %s", e.what());
        }
    }

    /*
    void publish_odom(const rclcpp::Time & stamp) {
        ...
    }
    */

    // rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture cap_;

    // cv::Ptr<cv::ORB> orb_;
    // cv::Ptr<cv::DescriptorMatcher> matcher_;
    // cv::Mat prev_frame_, prev_desc_;
    // std::vector<cv::KeyPoint> prev_kp_;
    // double total_theta_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VisualOdometryNode>());
    rclcpp::shutdown();
    return 0;
}
