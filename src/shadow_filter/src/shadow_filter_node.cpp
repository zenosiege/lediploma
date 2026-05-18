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

        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera", 10,
            std::bind(&ShadowFilterNode::image_callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/shadow_mask", 10);
        debug_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/debug_thresh", 10);

        RCLCPP_INFO(this->get_logger(), "Shadow Filter: Auto-exposure resistant mode enabled.");
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV Bridge error: %s", e.what());
            return;
        }

        cv::Mat frame = cv_ptr->image;
        if (frame.empty()) return;

        cv::Mat hsv, v_channel;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::extractChannel(hsv, v_channel, 2); // Берем только яркость

        // ==========================================
        // ШАГ 1: ДИНАМИЧЕСКИЙ ПОИСК ПОДЛОЖКИ (МЕТОД ОЦУ + CONVEX HULL)
        // ==========================================
        cv::Mat base_thresh;
        // Оцу находит границу между темным фоном и светлой подложкой
        cv::threshold(v_channel, base_thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        std::vector<std::vector<cv::Point>> base_contours;
        cv::findContours(base_thresh, base_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat dynamic_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        if (!base_contours.empty()) {
            // Ищем самую большую кляксу
            int largest_idx = 0;
            double max_base_area = 0;
            for (size_t i = 0; i < base_contours.size(); i++) {
                double area = cv::contourArea(base_contours[i]);
                if (area > max_base_area) {
                    max_base_area = area;
                    largest_idx = i;
                }
            }

            // --- НОВАЯ МАГИЯ ЗДЕСЬ ---
            // Вычисляем выпуклую оболочку для этого контура.
            // Даже если тень "прорвала" границу, Hull просто натянет "резинку"
            // поверх этого прорыва, создав сплошной диск.
            std::vector<cv::Point> hull;
            cv::convexHull(base_contours[largest_idx], hull);

            // Теперь рисуем не оригинальный контур, а эту выпуклую оболочку (и заливаем сплошь)
            std::vector<std::vector<cv::Point>> hull_contours;
            hull_contours.push_back(hull);
            cv::drawContours(dynamic_mask, hull_contours, 0, cv::Scalar(255), -1);

            // Сужаем маску (убираем грязные края)
            cv::Mat erode_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
            cv::erode(dynamic_mask, dynamic_mask, erode_kernel);
        }

        // ==========================================
        // ШАГ 2: УМНЫЙ ПОИСК ТЕНИ (АДАПТИВ)
        // ==========================================
        cv::Mat shadow_thresh;
        // Окно 51 пиксель справится с жирной тенью, константа 10 дает запас по контрасту
        cv::adaptiveThreshold(v_channel, shadow_thresh, 255,
                              cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv::THRESH_BINARY_INV, 51, 10);

        // ==========================================
        // ШАГ 3: НАЛОЖЕНИЕ МАСКИ И ДЕБАГ
        // ==========================================
        cv::Mat clean_shadow;
        // Оставляем от тени только то, что попало внутрь белого круга (dynamic_mask)
        cv::bitwise_and(shadow_thresh, dynamic_mask, clean_shadow);

        // Публикуем промежуточный результат (смотри в RViz топик /debug_thresh)
        std_msgs::msg::Header header = msg->header;
        cv_bridge::CvImage debug_msg(header, sensor_msgs::image_encodings::MONO8, clean_shadow);
        debug_publisher_->publish(*debug_msg.toImageMsg());

        // ==========================================
        // ШАГ 4: МОРФОЛОГИЯ И ФИНАЛЬНЫЙ ФИЛЬТР
        // ==========================================
        // Слегка чистим шум
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(clean_shadow, clean_shadow, cv::MORPH_OPEN, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(clean_shadow, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat final_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        double min_area = this->get_parameter("min_contour_area").as_double();
        double max_area = this->get_parameter("max_contour_area").as_double();

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            // Если пятно подходит по размеру — рисуем его в финальную маску
            if (area > min_area && area < max_area) {
                cv::drawContours(final_mask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), -1);
            }
        }

        // Публикуем готовую маску тени
        cv_bridge::CvImage out_msg(header, sensor_msgs::image_encodings::MONO8, final_mask);
        publisher_->publish(*out_msg.toImageMsg());
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_publisher_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ShadowFilterNode>());
    rclcpp::shutdown();
    return 0;
}
