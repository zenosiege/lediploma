#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

class CircleDetectorNode : public rclcpp::Node {
public:
    CircleDetectorNode() : Node("circle_detector_node") {
        this->declare_parameter("dp", 1.0);
        this->declare_parameter("min_dist", 60.0);
        this->declare_parameter("param1", 35.0);
        this->declare_parameter("param2", 50.0);
        this->declare_parameter("min_radius", 10);
        this->declare_parameter("max_radius", 500);
        this->declare_parameter("shadow_thresh", 80);

        // НОВЫЙ ПАРАМЕТР: во сколько раз уменьшаем зону поиска тени
        this->declare_parameter("radius_scale", 1.5);

        lastLogTime = this->now();

        imageSub = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera", 10,
            std::bind(&CircleDetectorNode::imageCallback, this, std::placeholders::_1));

        pointPub = this->create_publisher<geometry_msgs::msg::Point>("/circle_detected", 10);
        imagePub = this->create_publisher<sensor_msgs::msg::Image>("/image_processed", 10);

        RCLCPP_INFO(this->get_logger(), "Нода обновлена. Добавлено ограничение рабочей зоны.");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSub;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr pointPub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr imagePub;
    rclcpp::Time lastLogTime;

    double smoothX = 0.0;
    double smoothY = 0.0;
    double smoothR = 0.0;
    bool isFirstCircle = true;

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

        int shadowThresh = this->get_parameter("shadow_thresh").as_int();
        double radiusScale = this->get_parameter("radius_scale").as_double();

        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(imageGray, circles, cv::HOUGH_GRADIENT,
                         this->get_parameter("dp").as_double(),
                         this->get_parameter("min_dist").as_double(),
                         this->get_parameter("param1").as_double(),
                         this->get_parameter("param2").as_double(),
                         this->get_parameter("min_radius").as_int(),
                         this->get_parameter("max_radius").as_int());

        auto resultMsg = geometry_msgs::msg::Point();

        if (!circles.empty()) {
            cv::Vec3f largestCircle = circles[0];
            for (size_t i = 1; i < circles.size(); i++) {
                if (circles[i][2] > largestCircle[2]) largestCircle = circles[i];
            }

            if (isFirstCircle) {
                smoothX = largestCircle[0];
                smoothY = largestCircle[1];
                smoothR = largestCircle[2];
                isFirstCircle = false;
            } else {
                double alpha = 0.15;
                smoothX = smoothX * (1.0 - alpha) + largestCircle[0] * alpha;
                smoothY = smoothY * (1.0 - alpha) + largestCircle[1] * alpha;
                smoothR = smoothR * (1.0 - alpha) + largestCircle[2] * alpha;
            }

            cv::Point center(cvRound(smoothX), cvRound(smoothY));
            int radius = cvRound(smoothR);

            // ВЫЧИСЛЯЕМ РАДИУС ЗОНЫ ПОИСКА (отсекаем стенки)
            int searchRadius = cvRound(smoothR / radiusScale);

            // Создаем маску круга, но уже с уменьшенным searchRadius
            cv::Mat circleMask = cv::Mat::zeros(imageGray.size(), CV_8UC1);
            cv::circle(circleMask, center, searchRadius, cv::Scalar(255), -1);

            cv::Mat darkPixels;
            cv::threshold(imageGray, darkPixels, shadowThresh, 255, cv::THRESH_BINARY_INV);

            cv::Mat shadowOnly;
            cv::bitwise_and(darkPixels, circleMask, shadowOnly);

            // Вырезаем саму зубочистку из маски (черный круг в центре)
            cv::circle(shadowOnly, center, 15, cv::Scalar(0), -1);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(shadowOnly, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            if (!contours.empty()) {
                double maxArea = 0.0;
                int maxIdx = -1;

                for (size_t i = 0; i < contours.size(); i++) {
                    double area = cv::contourArea(contours[i]);
                    if (area > maxArea) {
                        maxArea = area;
                        maxIdx = (int)i;
                    }
                }

                if (maxIdx != -1 && maxArea > 50.0) {
                    cv::Moments m = cv::moments(contours[maxIdx]);
                    if (m.m00 > 0) {
                        int shadowX = cvRound(m.m10 / m.m00);
                        int shadowY = cvRound(m.m01 / m.m00);
                        cv::Point shadowCenter(shadowX, shadowY);

                        double angleRad = std::atan2(shadowY - center.y, shadowX - center.x);
                        double angleDeg = angleRad * 180.0 / M_PI;

                        // Линию тени рисуем длинной (до внешнего края стакана) для красоты
                        cv::Point lineEnd;
                        lineEnd.x = center.x + radius * std::cos(angleRad);
                        lineEnd.y = center.y + radius * std::sin(angleRad);

                        cv::line(cvPtr->image, center, lineEnd, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
                        cv::circle(cvPtr->image, shadowCenter, 5, cv::Scalar(0, 255, 255), -1);

                        auto currentTime = this->now();
                        if ((currentTime - lastLogTime).seconds() > 1.0) {
                            RCLCPP_INFO(this->get_logger(), "Угол: %.2f° | Полный R: %d px | Зона поиска: %d px", angleDeg, radius, searchRadius);
                            lastLogTime = currentTime;
                        }

                        resultMsg.x = center.x;
                        resultMsg.y = center.y;
                        resultMsg.z = angleDeg;
                    }

                    cv::drawContours(cvPtr->image, contours, maxIdx, cv::Scalar(0, 0, 255), -1);
                }
            }

            // Рисуем зеленый контур самого стакана (внешняя граница)
            cv::circle(cvPtr->image, center, radius, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            // Рисуем жёлтый контур рабочей зоны (отсекающий стенки)
            cv::circle(cvPtr->image, center, searchRadius, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
            // Центр
            cv::circle(cvPtr->image, center, 3, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
        }

        pointPub->publish(resultMsg);
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
