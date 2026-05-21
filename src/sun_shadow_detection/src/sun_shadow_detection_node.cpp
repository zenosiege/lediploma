#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

class ShadowTrackerNode : public rclcpp::Node {
public:
    ShadowTrackerNode() : Node("shadow_tracker_node") {
        // Пороги для новой картинки: белая тень на черном фоне
        this->declare_parameter("white_thresh", 200); // Всё светлее 200 - это тень
        this->declare_parameter("black_thresh", 50);  // Всё темнее 50 - это круг

        // Коэффициент обрезки краев (0.9 означает, что мы ищем тень только в пределах 90% радиуса)
        this->declare_parameter("edge_margin", 0.9);

        lastLogTime = this->now();

        imageSub = this->create_subscription<sensor_msgs::msg::Image>(
            "/shadow_mask_color", 10,
            std::bind(&ShadowTrackerNode::imageCallback, this, std::placeholders::_1));

        pointPub = this->create_publisher<geometry_msgs::msg::Point>("/circle_detected", 10);
        imagePub = this->create_publisher<sensor_msgs::msg::Image>("/image_processed", 10);

        RCLCPP_INFO(this->get_logger(), "Нода работает: БЕЛАЯ тень на ЧЕРНОМ круге. Защита от мусора включена.");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSub;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr pointPub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr imagePub;
    rclcpp::Time lastLogTime;

    // Переменные для сглаживания (чтобы круг не дергался)
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

        int whiteThresh = this->get_parameter("white_thresh").as_int();
        int blackThresh = this->get_parameter("black_thresh").as_int();
        double edgeMargin = this->get_parameter("edge_margin").as_double();

        // 2. ВЫДЕЛЕНИЕ ЦВЕТОВ
        cv::Mat shadowMask, blackCircleMask, fullCircleMask;

        // Белая тень
        cv::inRange(cvPtr->image, cv::Scalar(whiteThresh, whiteThresh, whiteThresh),
                    cv::Scalar(255, 255, 255), shadowMask);

        // Чёрный круг
        cv::inRange(cvPtr->image, cv::Scalar(0, 0, 0),
                    cv::Scalar(blackThresh, blackThresh, blackThresh), blackCircleMask);

        // Склеиваем, чтобы получить сплошной круг (без дырки от белой тени)
        cv::bitwise_or(blackCircleMask, shadowMask, fullCircleMask);

        auto resultMsg = geometry_msgs::msg::Point();

        // 3. ПОИСК ЦЕНТРА КРУГА И РАДИУСА
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

                // === СГЛАЖИВАНИЕ ДРОЖАНИЯ ===
                if (isFirstCircle) {
                    smoothX = centerFloat.x;
                    smoothY = centerFloat.y;
                    smoothR = radiusFloat;
                    isFirstCircle = false;
                } else {
                    double alpha = 0.15; // Плавность
                    smoothX = smoothX * (1.0 - alpha) + centerFloat.x * alpha;
                    smoothY = smoothY * (1.0 - alpha) + centerFloat.y * alpha;
                    smoothR = smoothR * (1.0 - alpha) + radiusFloat * alpha;
                }

                cv::Point center(cvRound(smoothX), cvRound(smoothY));
                int radius = cvRound(smoothR);

                // === ФИЛЬТРАЦИЯ БЕЛОГО МУСОРА ПО КРАЯМ ===
                // Создаем маску рабочей зоны, отсекая края
                int searchRadius = cvRound(smoothR * edgeMargin);
                cv::Mat searchMask = cv::Mat::zeros(shadowMask.size(), CV_8UC1);
                cv::circle(searchMask, center, searchRadius, cv::Scalar(255), -1);

                // Оставляем только те белые пиксели (тень), которые попали в searchMask
                cv::Mat cleanShadowMask;
                cv::bitwise_and(shadowMask, searchMask, cleanShadowMask);

                // 4. ПОИСК ТЕНИ
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

                            double dx = shadowX - center.x;
                            double dy = shadowY - center.y;
                            double angleRad = std::atan2(dx, -dy);
                            double angleDeg = angleRad * 180.0 / M_PI;

                            if (angleDeg < 0) angleDeg += 360.0;

                            cv::Point lineEnd;
                            lineEnd.x = center.x + radius * std::cos(angleRad - M_PI/2);
                            lineEnd.y = center.y + radius * std::sin(angleRad - M_PI/2);

                            // Рисуем
                            cv::line(cvPtr->image, center, lineEnd, cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
                            cv::circle(cvPtr->image, shadowCenter, 5, cv::Scalar(0, 255, 255), -1);

                            auto currentTime = this->now();
                            if ((currentTime - lastLogTime).seconds() > 1.0) {
                                RCLCPP_INFO(this->get_logger(), "Азимут тени: %.2f°", angleDeg);
                                lastLogTime = currentTime;
                            }

                            resultMsg.x = center.x;
                            resultMsg.y = center.y;
                            resultMsg.z = angleDeg;
                        }
                    }
                }

                // Визуализация
                cv::circle(cvPtr->image, center, radius, cv::Scalar(0, 255, 0), 2, cv::LINE_AA); // Контур круга
                cv::circle(cvPtr->image, center, searchRadius, cv::Scalar(0, 255, 255), 1, cv::LINE_AA); // Жёлтая рабочая зона
                cv::circle(cvPtr->image, center, 4, cv::Scalar(255, 0, 0), -1, cv::LINE_AA); // Центр
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
    rclcpp::spin(std::make_shared<ShadowTrackerNode>());
    rclcpp::shutdown();
    return 0;
}
