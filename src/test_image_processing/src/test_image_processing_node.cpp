#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>

class CircleDetectorNode : public rclcpp::Node {
public:
    CircleDetectorNode() : Node("circle_detector_node") {
        // Параметры для настройки на лету
        this->declare_parameter("dp", 1.0);
        this->declare_parameter("min_dist", 60.0);
        this->declare_parameter("param1", 35.0);
        this->declare_parameter("param2", 50.0);
        this->declare_parameter("min_radius", 10);
        this->declare_parameter("max_radius", 500);

        // Подписка на сырую камеру (чистый sensor_msgs без image_transport)
        imageSub = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera", 10,
            std::bind(&CircleDetectorNode::imageCallback, this, std::placeholders::_1));

        // Паблишеры: один для координат, второй для итоговой картинки
        pointPub = this->create_publisher<geometry_msgs::msg::Point>("/circle_detected", 10);
        imagePub = this->create_publisher<sensor_msgs::msg::Image>("/image_processed", 10);

        RCLCPP_INFO(this->get_logger(), "Нода запущена. Жду кадры с /camera...");
    }

private:
    // Нормальные человеческие названия без подчеркиваний на конце
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSub;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr pointPub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr imagePub;

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
        cv_bridge::CvImagePtr cvPtr;
        try {
            cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Ошибка cv_bridge: %s", e.what());
            return;
        }

        cv::Mat imageGray;
        cv::cvtColor(cvPtr->image, imageGray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(imageGray, imageGray, cv::Size(9, 9), 2, 2);

        // Читаем параметры
        double dp = this->get_parameter("dp").as_double();
        double minDist = this->get_parameter("min_dist").as_double();
        double param1 = this->get_parameter("param1").as_double();
        double param2 = this->get_parameter("param2").as_double();
        int minRadius = this->get_parameter("min_radius").as_int();
        int maxRadius = this->get_parameter("max_radius").as_int();

        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(imageGray, circles, cv::HOUGH_GRADIENT,
                         dp, minDist, param1, param2, minRadius, maxRadius);

        auto resultMsg = geometry_msgs::msg::Point();

        if (!circles.empty()) {
            cv::Vec3f largestCircle = circles[0];
            for (size_t i = 1; i < circles.size(); i++) {
                if (circles[i][2] > largestCircle[2]) {
                    largestCircle = circles[i];
                }
            }

            resultMsg.x = largestCircle[0];
            resultMsg.y = largestCircle[1];
            resultMsg.z = largestCircle[2];

            // Рисуем круги на кадре для визуализации
            cv::Point center(cvRound(largestCircle[0]), cvRound(largestCircle[1]));
            int radius = cvRound(largestCircle[2]);

            // Зеленый контур
            cv::circle(cvPtr->image, center, radius, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
            // Красный центр
            cv::circle(cvPtr->image, center, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        } else {
            resultMsg.x = 0.0;
            resultMsg.y = 0.0;
            resultMsg.z = 0.0;
        }

        // Шлем координаты
        pointPub->publish(resultMsg);

        // Конвертируем обратно в ROS-сообщение и шлем готовую картинку в топик
        sensor_msgs::msg::Image::SharedPtr outImageMsg =
            cv_bridge::CvImage(msg->header, "bgr8", cvPtr->image).toImageMsg();
        imagePub->publish(*outImageMsg);
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CircleDetectorNode>());
    rclcpp::shutdown();
    return 0;
}
