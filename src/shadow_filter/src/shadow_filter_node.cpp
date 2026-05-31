#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

class ShadowFilterNode : public rclcpp::Node
{
public:
    ShadowFilterNode() : Node("shadow_filter_node")
    {
        this->declare_parameter("min_contour_area", 300.0);
        this->declare_parameter("max_contour_area", 50000.0);
        this->declare_parameter("radius_shrink_ratio", 0.82);
        // Насколько тень должна быть резче (контрастнее) фона (от 3 до 15)
        this->declare_parameter("shadow_contrast", 6);
        // Размер огромного размытия для получения чистого фона (должен быть больше толщины тени)
        this->declare_parameter("bg_blur_size", 81);

        // Параметр для пространственного фильтра: насколько далеко от центра может начинаться тень (в долях от радиуса)
        this->declare_parameter("center_touch_tolerance", 0.3);

        imageSub = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera", 10,
            std::bind(&ShadowFilterNode::imageCallback, this, std::placeholders::_1));

        imagePub = this->create_publisher<sensor_msgs::msg::Image>("/shadow_mask", 10);
        debugPub = this->create_publisher<sensor_msgs::msg::Image>("/debug_thresh", 10);
        colorMaskPub = this->create_publisher<sensor_msgs::msg::Image>("/shadow_mask_color", 10);

        RCLCPP_INFO(this->get_logger(), "Shadow Filter V4: Анти-тряска (EMA) и Пространственный фильтр активированы.");
    }

private:
    // Переменные для хранения сглаженных координат между кадрами
    cv::Point2f smoothedCenter{-1.0f, -1.0f};
    float smoothedRadius = -1.0f;
    const float alpha = 0.1f; // Коэффициент плавности (меньше = плавнее, но инертнее)

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
        cv::extractChannel(hsv, vChannel, 2);

        double shrinkRatio = this->get_parameter("radius_shrink_ratio").as_double();
        double centerTolerance = this->get_parameter("center_touch_tolerance").as_double();


        // ==========================================
        // ШАГ 1: ПОИСК КРУГА И СГЛАЖИВАНИЕ (EMA)
        // ==========================================
        cv::Mat normalizedV, blurredBase, baseThresh;
        cv::normalize(vChannel, normalizedV, 0, 255, cv::NORM_MINMAX);
        cv::GaussianBlur(normalizedV, blurredBase, cv::Size(21, 21), 0);
        cv::threshold(blurredBase, baseThresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        std::vector<std::vector<cv::Point>> baseContours;
        cv::findContours(baseThresh, baseContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat dynamicMask = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::Point centerPoint(frame.cols / 2, frame.rows / 2);
        int finalRadius = std::min(frame.cols, frame.rows) / 4;

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

            cv::Point2f currentCenter;
            float currentRadius;
            cv::minEnclosingCircle(baseContours[largestIdx], currentCenter, currentRadius);

            // Применяем фильтр EMA для устранения тряски
            if (smoothedRadius < 0) { // Инициализация на первом кадре
                smoothedCenter = currentCenter;
                smoothedRadius = currentRadius;
            } else {
                smoothedCenter = alpha * currentCenter + (1.0f - alpha) * smoothedCenter;
                smoothedRadius = alpha * currentRadius + (1.0f - alpha) * smoothedRadius;
            }

            centerPoint.x = cvRound(smoothedCenter.x);
            centerPoint.y = cvRound(smoothedCenter.y);
            finalRadius = cvRound(smoothedRadius * shrinkRatio);

            cv::circle(dynamicMask, centerPoint, finalRadius, cv::Scalar(255), -1);
        }

        // ==========================================
        // ШАГ 2: АГРЕССИВНЫЙ ДЕНОЙЗ И ВЫЧИТАНИЕ ФОНА
        // ==========================================
        int shadowContrast = this->get_parameter("shadow_contrast").as_int();
        int bgBlurSize = this->get_parameter("bg_blur_size").as_int();
        if (bgBlurSize % 2 == 0) bgBlurSize += 1;

        cv::Mat denoisedV, bg, diff, shadowThresh;

        // 1. ПОДГОТОВКА ИЗОБРАЖЕНИЯ
        // Используем медианный фильтр с жирным окном (например, 7x7 или 9x9).
        // Он убьет весь математический шум матрицы, оставив гладкий пластик и резкую тень.
        cv::medianBlur(vChannel, denoisedV, 13);

        // Для дополнительной страховки от микро-теней полируем легким Гауссом
        cv::GaussianBlur(denoisedV, denoisedV, cv::Size(3, 3), 0);

        // 2. ВЫДЕЛЕНИЕ ФОНА (Низкочастотная составляющая)
        // Размываем УЖЕ ОЧИЩЕННУЮ картинку огромным ядром
        cv::GaussianBlur(denoisedV, bg, cv::Size(bgBlurSize, bgBlurSize), 0);

        // 3. ВЫЧИТАНИЕ (Оставляем только полезный сигнал)
        cv::subtract(bg, denoisedV, diff, dynamicMask);

        // 4. БИНАРИЗАЦИЯ
        cv::threshold(diff, shadowThresh, shadowContrast, 255, cv::THRESH_BINARY);

        cv::Mat cleanShadow = shadowThresh.clone();

        // ==========================================
        // ШАГ 3: АГРЕССИВНАЯ МОРФОЛОГИЯ И ФИЛЬТРЫ
        // ==========================================
        // УВЕЛИЧИЛИ Close: ядро 15x15 намертво склеит разорванные островки тени в одну линию
        cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
        cv::morphologyEx(cleanShadow, cleanShadow, cv::MORPH_CLOSE, closeKernel);

        // УМЕНЬШИЛИ Open: ядро 3x3 уберет только пиксельный шум, но не сожрет тонкую тень
        cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(cleanShadow, cleanShadow, cv::MORPH_OPEN, openKernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(cleanShadow, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat finalMask = cv::Mat::zeros(frame.size(), CV_8UC1);
        double minArea = this->get_parameter("min_contour_area").as_double();
        double maxArea = this->get_parameter("max_contour_area").as_double();
        double maxDistToGnomon = finalRadius * centerTolerance;

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);

            if (area > minArea && area < maxArea) {
                bool originatesFromCenter = false;

                for (const auto& pt : contour) {
                    double dist = cv::norm(cv::Point2f(pt.x, pt.y) - cv::Point2f(centerPoint.x, centerPoint.y));
                    if (dist < maxDistToGnomon) {
                        originatesFromCenter = true;
                        break;
                    }
                }

                if (originatesFromCenter) {
                    cv::drawContours(finalMask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), -1);
                }
            }
        }

        std_msgs::msg::Header header = msg->header;
        cv_bridge::CvImage debugMsg(header, sensor_msgs::image_encodings::MONO8, cleanShadow);
        debugPub->publish(*debugMsg.toImageMsg());
        cv_bridge::CvImage outMsg(header, sensor_msgs::image_encodings::MONO8, finalMask);
        imagePub->publish(*outMsg.toImageMsg());

        // ==========================================
        // ШАГ 4: ТРЕХЦВЕТНАЯ КАРТИНКА
        // ==========================================
        cv::Mat colorMask = cv::Mat::zeros(frame.size(), CV_8UC3);
        colorMask.setTo(cv::Scalar(0, 0, 255));
        cv::circle(colorMask, centerPoint, finalRadius, cv::Scalar(0, 0, 0), -1);
        colorMask.setTo(cv::Scalar(255, 255, 255), finalMask);

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
