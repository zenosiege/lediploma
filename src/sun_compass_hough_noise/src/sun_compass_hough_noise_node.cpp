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
        // Объявляем параметры камеры
        this->declare_parameter("camera_id", 0);
        this->declare_parameter("frame_width", 640);
        this->declare_parameter("frame_height", 480);

        // === ПАРАМЕТРЫ ШУМА ДЛЯ ТЕСТИРОВАНИЯ ===
        this->declare_parameter("enable_noise", true); // Включить/выключить шум
        this->declare_parameter("noise_intensity", 30.0); // Сила шума (от 5 до 100)

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
                        "Камера %d открыта, разрешение: %dx%d (Зеркалирование и генератор шума готовы)",
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
            // Отражаем кадр по горизонтали
            cv::Mat flipped_frame;
            cv::flip(current_frame, flipped_frame, 1);
            current_frame = flipped_frame;

            // === ГЕНЕРАЦИЯ ШУМА ===
            bool enableNoise = this->get_parameter("enable_noise").as_bool();
            if (enableNoise) {
                double noiseStdDev = this->get_parameter("noise_intensity").as_double();

                // Создаем пустую матрицу под шум в формате 32-битных флоатов (для отрицательных значений)
                cv::Mat noise(current_frame.size(), CV_32FC3);

                // Генерируем Гауссовский шум (среднее 0, отклонение = noiseStdDev)
                cv::randn(noise, 0.0, noiseStdDev);

                // Чтобы OpenCV корректно сложил шум с картинкой без артефактов переполнения,
                // конвертируем текущий кадр во флоаты, добавляем шум и возвращаем обратно в 8 бит
                cv::Mat float_frame;
                current_frame.convertTo(float_frame, CV_32FC3);
                float_frame += noise;
                float_frame.convertTo(current_frame, CV_8UC3);
            }

            // Показываем изображение для отладки
            cv::imshow("Raw Camera Frame", current_frame);
            cv::waitKey(1);

            // Конвертируем OpenCV Mat в ROS2 Image message
            sensor_msgs::msg::Image::SharedPtr ros_image =
                cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", current_frame).toImageMsg();

            // Заполняем метаданные
            ros_image->header.frame_id = "camera_optical_frame";
            ros_image->header.stamp = this->now();

            // Публикуем зашумленное изображение в топик
            image_pub_.publish(ros_image);

        } catch (cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "OpenCV error: %s", e.what());
        }
    }

    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture cap_;
    image_transport::Publisher image_pub_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VisualOdometryNode>());
    rclcpp::shutdown();
    return 0;
}
