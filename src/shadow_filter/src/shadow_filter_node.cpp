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
        // Параметры для настройки "на лету"
        this->declare_parameter("min_contour_area", 500.0);
        this->declare_parameter("max_contour_area", 50000.0);

        // Подписываемся на сырую картинку (замени топик на свой)
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            std::bind(&ShadowFilterNode::image_callback, this, std::placeholders::_1));

        // Публикуем готовую черно-белую маску тени
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/shadow_mask", 10);

        RCLCPP_INFO(this->get_logger(), "Shadow Filter Node initialized. Ready to clean the mess.");
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            // Перегоняем ROS-сообщение в формат OpenCV
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat frame = cv_ptr->image;
        if (frame.empty()) return;

        // 1. Физическая обрезка зоны (ROI)
        // Вычисляем центр и радиус (в 1.5 раза меньше от края, чтобы отсечь стенки)
        cv::Point center(frame.cols / 2, frame.rows / 2);
        int max_radius = std::min(frame.cols, frame.rows) / 2;
        int roi_radius = max_radius / 1.5;

        cv::Mat roi_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::circle(roi_mask, center, roi_radius, cv::Scalar(255), -1);

        cv::Mat frame_roi;
        frame.copyTo(frame_roi, roi_mask);

        // 2. Уход от RGB: переводим в HSV и берем канал яркости (Value)
        cv::Mat hsv, v_channel;
        cv::cvtColor(frame_roi, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsv_channels;
        cv::split(hsv, hsv_channels);
        v_channel = hsv_channels[2]; // Канал V

        // 3. Умная бинаризация (Адаптивный порог)
        // Тень темная, поэтому используем THRESH_BINARY_INV, чтобы она стала белой на черном фоне
        cv::Mat thresh;
        cv::adaptiveThreshold(v_channel, thresh, 255,
                              cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv::THRESH_BINARY_INV, 21, 15);

        // Применяем нашу круглую маску еще раз, чтобы убрать мусор,
        // который адаптивный порог мог вытянуть за пределами круга
        cv::bitwise_and(thresh, roi_mask, thresh);

        // 4. Выкашивание шумов (Морфология: Открытие)
        // Эрозия убьет мелкий шум, дилатация вернет толщину нормальной тени
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel);

        // 5. Фильтрация контуров
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat final_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        double min_area = this->get_parameter("min_contour_area").as_double();
        double max_area = this->get_parameter("max_contour_area").as_double();

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > min_area && area < max_area) {
                // Проверяем точку роста: тень должна исходить от гномона (центра кадра)
                // Для простоты проверяем, пересекает ли bounding box центр
                cv::Rect bbox = cv::boundingRect(contour);
                if (bbox.contains(center)) {
                    // Рисуем отфильтрованную тень сплошным белым цветом
                    cv::drawContours(final_mask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), -1);
                }
            }
        }

        // Пакуем обратно в ROS-сообщение и публикуем
        std_msgs::msg::Header header = msg->header;
        cv_bridge::CvImage out_msg(header, sensor_msgs::image_encodings::MONO8, final_mask);
        publisher_->publish(*out_msg.toImageMsg());
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ShadowFilterNode>());
    rclcpp::shutdown();
    return 0;
}

