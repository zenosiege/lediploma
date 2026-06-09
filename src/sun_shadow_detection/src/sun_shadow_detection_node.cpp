#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

class SunShadowDetectionNode : public rclcpp::Node {
public:
    SunShadowDetectionNode() : Node("sun_shadow_detection_node") {
        // Настройки зрения
        this->declare_parameter("white_thresh", 200);
        this->declare_parameter("black_thresh", 50);
        this->declare_parameter("edge_margin", 0.9);

        // Географические координаты (по умолчанию - Рязань)
        this->declare_parameter("latitude", 54.6269);
        this->declare_parameter("longitude", 39.7145);

        lastLogTime = this->now();

        imageSub = this->create_subscription<sensor_msgs::msg::Image>(
            "/shadow_mask_color", 10,
            std::bind(&SunShadowDetectionNode::imageCallback, this, std::placeholders::_1));

        pointPub = this->create_publisher<geometry_msgs::msg::Point>("/robot_heading", 10);
        imagePub = this->create_publisher<sensor_msgs::msg::Image>("/image_processed", 10);

        RCLCPP_INFO(this->get_logger(), "Астрономический навигатор запущен. Вычисляем истинный курс...");
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

    // === АСТРОНОМИЧЕСКИЙ БЛОК ===
    // Вычисляет азимут Солнца на основе времени и координат
    double getSunAzimuth(double lat, double lon, double unix_time) {
        // Перевод времени в дни с эпохи J2000
        double d = (unix_time / 86400.0) + 2440587.5 - 2451545.0;

        // Эклиптические координаты
        double L = fmod(280.460 + 0.9856474 * d, 360.0);
        if (L < 0) L += 360.0;
        double g = fmod(357.528 + 0.9856003 * d, 360.0);
        if (g < 0) g += 360.0;

        double g_rad = g * M_PI / 180.0;
        double lambda = L + 1.915 * std::sin(g_rad) + 0.020 * std::sin(2 * g_rad);
        double lambda_rad = lambda * M_PI / 180.0;

        double epsilon_rad = (23.439 - 0.0000004 * d) * M_PI / 180.0;

        // Склонение и прямое восхождение
        double delta_rad = std::asin(std::sin(epsilon_rad) * std::sin(lambda_rad));
        double alpha_rad = std::atan2(std::cos(epsilon_rad) * std::sin(lambda_rad), std::cos(lambda_rad));

        // Звездное время
        double gmst = fmod(6.697375 + 0.0657098242 * d + (fmod(unix_time, 86400.0) / 3600.0) * 1.0027379, 24.0);
        if (gmst < 0) gmst += 24.0;

        double lmst_rad = (gmst * 15.0 + lon) * M_PI / 180.0;
        double H_rad = lmst_rad - alpha_rad;
        double lat_rad = lat * M_PI / 180.0;

        // Расчет азимута
        double az_rad = std::atan2(std::sin(H_rad), std::cos(H_rad) * std::sin(lat_rad) - std::tan(delta_rad) * std::cos(lat_rad));
        double az_deg = az_rad * 180.0 / M_PI;

        // Переводим отсчет от Севера в формат 0..360
        az_deg = fmod(az_deg + 180.0, 360.0);
        return az_deg;
    }

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
        cv_bridge::CvImagePtr cvPtr;
        try {
            cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            return;
        }

        // Берем параметры
        int whiteThresh = this->get_parameter("white_thresh").as_int();
        int blackThresh = this->get_parameter("black_thresh").as_int();
        double edgeMargin = this->get_parameter("edge_margin").as_double();
        double lat = this->get_parameter("latitude").as_double();
        double lon = this->get_parameter("longitude").as_double();

        cv::Mat shadowMask, blackCircleMask, fullCircleMask;
        cv::inRange(cvPtr->image, cv::Scalar(whiteThresh, whiteThresh, whiteThresh), cv::Scalar(255, 255, 255), shadowMask);
        cv::inRange(cvPtr->image, cv::Scalar(0, 0, 0), cv::Scalar(blackThresh, blackThresh, blackThresh), blackCircleMask);
        cv::bitwise_or(blackCircleMask, shadowMask, fullCircleMask);

        auto resultMsg = geometry_msgs::msg::Point();

        std::vector<std::vector<cv::Point>> circleContours;
        cv::findContours(fullCircleMask, circleContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (!circleContours.empty()) {
            double maxCircleArea = 0.0;
            int mainCircleIdx = -1;
            for (size_t i = 0; i < circleContours.size(); i++) {
                double area = cv::contourArea(circleContours[i]);
                if (area > maxCircleArea) {
                    maxCircleArea = area;
                    mainCircleIdx = (int)i;
                }
            }

            if (mainCircleIdx != -1 && maxCircleArea > 1000) {
                cv::Point2f centerFloat;
                float radiusFloat;
                cv::minEnclosingCircle(circleContours[mainCircleIdx], centerFloat, radiusFloat);

                if (isFirstCircle) {
                    smoothX = centerFloat.x;
                    smoothY = centerFloat.y;
                    smoothR = radiusFloat;
                    isFirstCircle = false;
                } else {
                    double alpha = 0.15;
                    smoothX = smoothX * (1.0 - alpha) + centerFloat.x * alpha;
                    smoothY = smoothY * (1.0 - alpha) + centerFloat.y * alpha;
                    smoothR = smoothR * (1.0 - alpha) + radiusFloat * alpha;
                }

                cv::Point center(cvRound(smoothX), cvRound(smoothY));
                int radius = cvRound(smoothR);

                int searchRadius = cvRound(smoothR * edgeMargin);
                cv::Mat searchMask = cv::Mat::zeros(shadowMask.size(), CV_8UC1);
                cv::circle(searchMask, center, searchRadius, cv::Scalar(255), -1);

                cv::Mat cleanShadowMask;
                cv::bitwise_and(shadowMask, searchMask, cleanShadowMask);

                std::vector<std::vector<cv::Point>> shadowContours;
                cv::findContours(cleanShadowMask, shadowContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                if (!shadowContours.empty()) {
                    double maxShadowArea = 0.0;
                    int mainShadowIdx = -1;

                    for (size_t i = 0; i < shadowContours.size(); i++) {
                        double area = cv::contourArea(shadowContours[i]);
                        if (area > maxShadowArea) {
                            maxShadowArea = area;
                            mainShadowIdx = (int)i;
                        }
                    }

                    if (mainShadowIdx != -1 && maxShadowArea > 20) {
                        cv::Moments m = cv::moments(shadowContours[mainShadowIdx]);
                        if (m.m00 > 0) {
                            int shadowX = cvRound(m.m10 / m.m00);
                            int shadowY = cvRound(m.m01 / m.m00);
                            cv::Point shadowCenter(shadowX, shadowY);

                            // 1. Считаем локальный угол тени в камере
                            double dx = shadowX - center.x;
                            double dy = shadowY - center.y;
                            double shadowAngle = std::atan2(dx, -dy) * 180.0 / M_PI;
                            if (shadowAngle < 0) shadowAngle += 360.0;

                            // 2. Считаем Азимут Солнца по времени
                            double unix_time = this->now().seconds();
                            double sunAzimuth = getSunAzimuth(lat, lon, unix_time);

                            // 3. ВЫЧИСЛЯЕМ ИСТИННЫЙ КУРС РОБОТА
                            double robotHeading = fmod(sunAzimuth + 180.0 - shadowAngle + 360.0, 360.0);

                            // Отрисовка
                            cv::Point lineEnd;
                            lineEnd.x = center.x + radius * std::cos((shadowAngle - 90) * M_PI / 180.0);
                            lineEnd.y = center.y + radius * std::sin((shadowAngle - 90) * M_PI / 180.0);

                            cv::line(cvPtr->image, center, lineEnd, cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
                            cv::circle(cvPtr->image, shadowCenter, 5, cv::Scalar(0, 255, 255), -1);

                            // Вывод всех данных в терминал (раз в секунду)
                            auto currentTime = this->now();
                            if ((currentTime - lastLogTime).seconds() > 1.0) {
                                RCLCPP_INFO(this->get_logger(),
                                            "\n--- Навигация ---\n"
                                            "Азимут Солнца: %.2f°\n"
                                            "Угол тени:     %.2f°\n"
                                            "КУРС РОБОТА:   %.2f°",
                                            sunAzimuth, shadowAngle, robotHeading);
                                lastLogTime = currentTime;
                            }

                            // Публикуем ИСТИННЫЙ КУРС в координату Z
                            resultMsg.x = center.x;
                            resultMsg.y = center.y;
                            resultMsg.z = robotHeading;
                        }
                    }
                }

                cv::circle(cvPtr->image, center, radius, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                cv::circle(cvPtr->image, center, searchRadius, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
                cv::circle(cvPtr->image, center, 4, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
            }
        }

        pointPub->publish(resultMsg);
        sensor_msgs::msg::Image::SharedPtr outImageMsg =
            cv_bridge::CvImage(msg->header, "bgr8", cvPtr->image).toImageMsg();
        imagePub->publish(*outImageMsg);
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SunShadowDetectionNode>());
    rclcpp::shutdown();
    return 0;
}
