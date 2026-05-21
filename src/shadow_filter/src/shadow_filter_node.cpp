#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>

class ShadowFilterNode : public rclcpp::Node
{
public:
    ShadowFilterNode() : Node("shadow_filter_node")
    {
        // Настройки размеров тени
        this->declare_parameter("min_contour_area", 20.0);
        this->declare_parameter("max_contour_area", 50000.0);

        imageSub = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera", 10,
            std::bind(&ShadowFilterNode::imageCallback, this, std::placeholders::_1));

        imagePub = this->create_publisher<sensor_msgs::msg::Image>("/shadow_mask", 10);
        debugPub = this->create_publisher<sensor_msgs::msg::Image>("/debug_thresh", 10);

        // Топик для Лёшеньки: выдает идеальную трехцветную геометрию
        colorMaskPub = this->create_publisher<sensor_msgs::msg::Image>("/shadow_mask_color", 10);

        RCLCPP_INFO(this->get_logger(), "Shadow Filter: Нода полностью готова. Стиль кода исправлен.");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv_bridge::CvImagePtr cvPtr;
        try {
            cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Ошибка CV Bridge: %s", e.what());
            return;
        }

        cv::Mat frame = cvPtr->image;
        if (frame.empty()) return;

        cv::Mat hsv, vChannel;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::extractChannel(hsv, vChannel, 2); // Берем только яркость

        // ==========================================
        // ШАГ 1: ДИНАМИЧЕСКИЙ ПОИСК ПОДЛОЖКИ (ОЦУ + CONVEX HULL)
        // ==========================================
        cv::Mat baseThresh;
        cv::threshold(vChannel, baseThresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        std::vector<std::vector<cv::Point>> baseContours;
        cv::findContours(baseThresh, baseContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat dynamicMask = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::Point centerPoint(frame.cols / 2, frame.rows / 2);
        int finalRadius = std::min(frame.cols, frame.rows) / 4; // Дефолтный радиус на случай

        if (!baseContours.empty()) {
            int largestIdx = 0;
            double maxBaseArea = 0;
            for (size_t i = 0; i < baseContours.size(); i++) {
                double area = cv::contourArea(baseContours[i]);
                if (area > maxBaseArea) {
                    maxBaseArea = area;
                    largestIdx = i;
                }
            }

            // Натягиваем выпуклую оболочку (Convex Hull), чтобы убрать вырезы от тени на краях
            std::vector<cv::Point> hull;
            cv::convexHull(baseContours[largestIdx], hull);

            // Описываем ИДЕАЛЬНЫЙ круг поверх нашей оболочки
            cv::Point2f centerF;
            float radiusF;
            cv::minEnclosingCircle(hull, centerF, radiusF);

            // Переводим в целочисленные параметры для рисования
            centerPoint.x = cvRound(centerF.x);
            centerPoint.y = cvRound(centerF.y);

            // Сужаем радиус маски на 10%, чтобы гарантированно отрезать грязные бортики стакана
            finalRadius = cvRound(radiusF * 0.90);

            // Рисуем на маску ИДЕАЛЬНЫЙ ровный геометрический круг
            cv::circle(dynamicMask, centerPoint, finalRadius, cv::Scalar(255), -1);
        }

        // ==========================================
        // ШАГ 2: УМНЫЙ ПОИСК ТЕНИ (АДАПТИВ)
        // ==========================================
        cv::Mat shadowThresh;
        cv::adaptiveThreshold(vChannel, shadowThresh, 255,
                              cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv::THRESH_BINARY_INV, 51, 10);

        // ==========================================
        // ШАГ 3: НАЛОЖЕНИЕ ИДЕАЛЬНОЙ МАСКИ КРУГА
        // ==========================================
        cv::Mat cleanShadow;
        cv::bitwise_and(shadowThresh, dynamicMask, cleanShadow);

        std_msgs::msg::Header header = msg->header;
        cv_bridge::CvImage debugMsg(header, sensor_msgs::image_encodings::MONO8, cleanShadow);
        debugPub->publish(*debugMsg.toImageMsg());

        // ==========================================
        // ШАГ 4: МОРФОЛОГИЯ И ФИНАЛЬНЫЙ ФИЛЬТР ТЕНИ
        // ==========================================
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(cleanShadow, cleanShadow, cv::MORPH_OPEN, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(cleanShadow, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat finalMask = cv::Mat::zeros(frame.size(), CV_8UC1);
        double minArea = this->get_parameter("min_contour_area").as_double();
        double maxArea = this->get_parameter("max_contour_area").as_double();

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > minArea && area < maxArea) {
                cv::drawContours(finalMask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), -1);
            }
        }

        cv_bridge::CvImage outMsg(header, sensor_msgs::image_encodings::MONO8, finalMask);
        imagePub->publish(*outMsg.toImageMsg());

        // ==========================================
        // ШАГ 5: ГЕНЕРАЦИЯ ИДЕАЛЬНОЙ ТРЕХЦВЕТНОЙ КАРТИНКИ
        // ==========================================
        cv::Mat colorMask = cv::Mat::zeros(frame.size(), CV_8UC3);

        // 1. Заливаем вообще всё полотно идеальным красным цветом (BGR: 0, 0, 255)
        colorMask.setTo(cv::Scalar(0, 0, 255));

        // 2. Чертим ИДЕАЛЬНЫЙ ровный черный круг по вычисленным ранее координатам центра и радиуса
        cv::circle(colorMask, centerPoint, finalRadius, cv::Scalar(0, 0, 0), -1);

        // 3. Штампуем поверх черного круга нашу чистую белую тень по трафарету finalMask
        colorMask.setTo(cv::Scalar(255, 255, 255), finalMask);

        // Публикуем идеальную трехцветную маску в топик для Алексея
        cv_bridge::CvImage colorOutMsg(header, sensor_msgs::image_encodings::BGR8, colorMask);
        colorMaskPub->publish(*colorOutMsg.toImageMsg());
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr imagePub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debugPub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr colorMaskPub;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ShadowFilterNode>());
    rclcpp::shutdown();
    return 0;
}
