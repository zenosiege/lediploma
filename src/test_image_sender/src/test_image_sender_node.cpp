#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>

using namespace std::chrono_literals;


// НОДА ОТПРАВКИ ИЗОБРАЖЕНИЯ

class TestImageSender : public rclcpp::Node
{
public:
    TestImageSender():
        Node("test_image_sender_node"),
        counter(0)
    {
        pub = image_transport::create_publisher(this, "/camera");
        image1 = cv::imread("/home/wattookie/Pictures/circles1.jpg");
        image2 = cv::imread("/home/wattookie/Pictures/circles2.jpg");
        timer = this->create_wall_timer(200ms, // 5 Гц = 200ms
                                        std::bind(&TestImageSender::timer_callback, this));
    }

private:
    image_transport::Publisher pub;
    cv::Mat image1, image2;
    rclcpp::TimerBase::SharedPtr timer;
    unsigned int counter;

    void timer_callback() {
        cv::Mat current_image = (counter % 2 == 0) ? image1 : image2;
        // ? - тернарный опператор. Если условие правда, то вернёт изображение до двоеточия, иначе - после

        counter++;

        // конвертация  изображение в сообщение типа Image, чтобы отправить его в топик /camera
        sensor_msgs::msg::Image::SharedPtr msg =
            cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", current_image).toImageMsg();
        msg->header.frame_id = "/camera_optical"; // фрейм или система координат
        msg->header.stamp = this->now();
        pub.publish(msg);
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TestImageSender>());
    rclcpp::shutdown();
    return 0;
}
