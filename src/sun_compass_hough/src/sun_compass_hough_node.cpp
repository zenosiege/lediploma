#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

using namespace std::chrono_literals;

class VisualOdometryNode : public rclcpp::Node {
public:
    VisualOdometryNode() : Node("visual_odom_node") {
        // Объявляем параметры
        this->declare_parameter("camera_id", 2);
        this->declare_parameter("frame_width", 640);
        this->declare_parameter("frame_height", 480);

        // Получаем значения параметров
        int camera_id = this->get_parameter("camera_id").as_int();
        int frame_width = this->get_parameter("frame_width").as_int();
        int frame_height = this->get_parameter("frame_height").as_int();

        // Открываем камеру
        cap_.open(camera_id, cv::CAP_V4L2);

        if (cap_.isOpened()) {
            cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
            cap_.set(cv::CAP_PROP_FRAME_WIDTH, frame_width);
            cap_.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height);
            // Устанавливаем формат MJPG для снижения нагрузки
            cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));

            RCLCPP_INFO(this->get_logger(),
                        "Камера %d открыта, разрешение: %dx%d",
                        camera_id, frame_width, frame_height);
        } else {
            RCLCPP_ERROR(this->get_logger(),
                         "Не удалось открыть камеру %d", camera_id);
            return;
        }

        // Создаём publisher для изображений
        image_pub_ = image_transport::create_publisher(this, "/camera");

        cv::namedWindow("Raw Camera Frame", cv::WINDOW_NORMAL);
        cv::resizeWindow("Raw Camera Frame", frame_width, frame_height);

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
            // Показываем изображение для отладки
            cv::imshow("Raw Camera Frame", current_frame);
            cv::waitKey(1);

            // Конвертируем OpenCV Mat в ROS2 Image message
            sensor_msgs::msg::Image::SharedPtr ros_image =
                cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", current_frame).toImageMsg();

            // Заполняем метаданные
            ros_image->header.frame_id = "camera_optical_frame";
            ros_image->header.stamp = this->now();

            // Публикуем изображение в топик
            image_pub_.publish(ros_image);

            /*
            ================================================================
               БЛОК ВИЗУАЛЬНОЙ ОДОМЕТРИИ ВРЕМЕННО ЗАКОММЕНТИРОВАН
            ================================================================
            std::vector<cv::KeyPoint> curr_kp;
            cv::Mat curr_desc;

            orb_->detectAndCompute(current_frame, cv::noArray(), curr_kp, curr_desc);
            if (curr_desc.empty()) return;

            if (!prev_desc_.empty()) {
                // ... весь код с RANSAC, вычислением углов и публикацией ...
            }

            prev_frame_ = current_frame.clone();
            prev_kp_ = curr_kp;
            prev_desc_ = curr_desc.clone();
            ================================================================
            */

        } catch (cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "OpenCV error: %s", e.what());
        }
    }

    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture cap_;
    image_transport::Publisher image_pub_;

    // Переменные для визуальной одометрии (закомментированы)
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
