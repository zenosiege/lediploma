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

        RCLCPP_INFO(this->get_logger(), "Shadow Filter включен.");
    }

private:
    // Переменные для хранения сглаженных координат между кадрами
    cv::Point2f smoothedCenter{-1.0f, -1.0f};
    float smoothedRadius = -1.0f;
    const float alpha = 0.1f; // Коэффициент плавности (меньше = плавнее, но инертнее)

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv_bridge::CvImagePtr cvPtr; // изображение, переданное через указатель
        try {
            // копируем в указатель изображение из сообщения в формате BGR8
            cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Ошибка CV Bridge: %s", e.what());
            return;
        }

        // поскольку cvPtr содержит не только кадр, но и его кодировку и заголовок,
        // то вытаскиваем из него изображение в отдельную переменную
        cv::Mat frame = cvPtr->image;
        if (frame.empty()) return;

        // ещё две матричные переменные
        // hsv - под изображение в новом формате HSV (Hue, Saturation, Value)
        // vChannel - под хранение ТОЛЬКО канала яркости (Value) исходное изображения
        cv::Mat hsv, vChannel;

        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV); // конвертация из BGR в HSV
        cv::extractChannel(hsv, vChannel, 2); // извлечение канала яркости в vChannel
        // (0 - Hue, 1 - Saturation, 2 - Value)

        double shrinkRatio = this->get_parameter("radius_shrink_ratio").as_double();
        double centerTolerance = this->get_parameter("center_touch_tolerance").as_double();


        // ==========================================
        // ШАГ 1: ПОИСК КРУГА И СГЛАЖИВАНИЕ (пока тень не ищем)
        // ==========================================
        cv::Mat normalizedV, blurredBase, baseThresh;

        // гауссовское размытие с нулевым средним и размером ядра 51х51
        // ядро специально большое, чтобы на выходе получилось - тёмный фон и просто круг
        // (усреднение с большим ядром сотрёт тень от гномона, ибо о ней и не стоит вопроса сейчас)
        cv::GaussianBlur(vChannel, blurredBase, cv::Size(51, 51), 0);

        // нормализация vChannel в диапазоне от 0 до 255 по алгоритму NORM_MINMAX
        // и передача результата в normalizedV
        // (растяжение некого диапазона на всю ширину, потому что некоторых уровней
        // яркостей, таких как абсолютно черный - 0, может и не быть. Нормализованное
        // изображение даёт больше информации для алгоритма, отчего он работает лучше)
        cv::normalize(blurredBase, normalizedV, 0, 255, cv::NORM_MINMAX);

        // делим картинку на чёрное(фон) и белое(экран стканчика).
        // размытое изображение оценивается методом Отцу, и получившееся значение
        // порога идёт в саму функцию бинаризации вместо указанного нуля
        // (поэтому THRESH_BINARY и THRESH_OTSU логически сложены)
        // на выходе получаем бинаризированное изображение baseThresh
        cv::threshold(blurredBase, baseThresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        // далее ищем контуры. Функция findContours требует массив массивов,
        // поскольку внутренний массив собирает в себе точки одной структуры (контура),
        // а внеший массив собирает в себе сами структуры.
        // -----------------------------------------------------------------------------
        // Режим извлечения контуров RETR_EXTERNAL находит только самые внешние контуры,
        // исключая области внутри контуров (наподобие отверстия в центре экрана гномона),
        // то есть теоретически, если cv::threshold выдал не совсем идеальную картину, где
        // внутри циферблата есть "дыры", то этот режим игнорирует их и учитывает только
        // САМУЮ внешнюю границу.
        // -----------------------------------------------------------------------------
        // Метод аппроксимации CHAIN_APPROX_SIMPLE позволяет сократить количество точек
        // у контура посредством аппроксимации горизонтальных, вертикальных и диагональных сегментов
        // контура, оставляя лишь их конечные точки.
        // Например, вместо того, чтобы кодировать прямоугольник 100 точками, достаточно 4 по углам.
        std::vector<std::vector<cv::Point>> baseContours;
        cv::findContours(baseThresh, baseContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // делаем пустую ч/б-картинку, соразмерную кадру с камеры (8U - 8 бит, C1 - одноканальный)
        cv::Mat dynamicMask = cv::Mat::zeros(frame.size(), CV_8UC1);

        // задаём центр циферблата "по умолчанию". В идеале, чтобы на практике
        // он (центр) находился там, где и центр кадра, потому что если алгоритму не
        // получится вычислить его реальный центр, то координаты центра кадра будут взяты за основу.
        cv::Point centerPoint(frame.cols / 2, frame.rows / 2);

        // аналогично, но с радиусом предполагаемого циферблата
        // (берётся меньшая из сторон кадра, поделённая на 4,
        // золотая середина своего рода, чтобы "идеальный" круг
        // циферблата помещался в экран)
        int finalRadius = std::min(frame.cols, frame.rows) / 4;

        // если всё хорошо и контур (или контуры) всё-таки есть
        if (!baseContours.empty()) {

            // берём самый большой по площади контур - им должен быть циферблат
            int largestIdx = 0;
            double maxBaseArea = 0;
            for (size_t i = 0; i < baseContours.size(); i++) {
                double area = cv::contourArea(baseContours[i]);
                if (area > maxBaseArea) {
                    maxBaseArea = area;
                    largestIdx = i;
                }
            }

            // строим Минимально Охватывающую Окружность по контуру с самой большой площадью
            // и берём её радиус и центр
            cv::Point2f currentCenter;
            float currentRadius;
            cv::minEnclosingCircle(baseContours[largestIdx], currentCenter, currentRadius);

            // в качестве ФНЧ применяем т.н. Экспоненциальное Скользящее Среднее (EMA)
            // математически прост - к взвешенному значению предыдущего сглаженного значения
            // радиуса прибавляется взвешенное найденное значение и берём это как новое сглаженное.
            // (сумма весов равна единице, от коэффициента alpha зависит резкость фильтра.
            // Чем меньше - тем плавнее).
            if (smoothedRadius < 0) { // инициализация на первом кадре
                smoothedCenter = currentCenter;
                smoothedRadius = currentRadius;
            } else {
                // c учётом того, что alpha = 0.1, то центр и радиус будут медленно сдвигаться к истинным,
                // но при 30 кадрах это происходит быстро. Больше частота обновления - быстрее сходится.
                smoothedCenter = alpha * currentCenter + (1.0f - alpha) * smoothedCenter;
                smoothedRadius = alpha * currentRadius + (1.0f - alpha) * smoothedRadius;
            }

            //округляем до ближайшего целого значения
            centerPoint.x = cvRound(smoothedCenter.x);
            centerPoint.y = cvRound(smoothedCenter.y);
            finalRadius = cvRound(smoothedRadius * shrinkRatio);
            // (урезаем немного радиус для того, чтобы быть уверенным в том,
            // что область принадлежит циферблату)

            // и наконец рисуем залитый белым круг в dynamicMask
            // (-1 в конце, она же cv::FILLED - это инструкция чтобы не просто провести контур,
            // но и залить его)
            cv::circle(dynamicMask, centerPoint, finalRadius, cv::Scalar(255), -1);
        }

        // ==========================================
        // ШАГ 2: ИСКЛЮЧЕНИЕ ШУМА И ВЫЧИТАНИЕ ФОНА
        // ==========================================
        int shadowContrast = this->get_parameter("shadow_contrast").as_int(); // фикс. порог
        int bgBlurSize = this->get_parameter("bg_blur_size").as_int();

        // для симметричного применения фильтра относительно центрального пикселя
        // необходимо нечётное число, это условие обеспечивает это.
        if (bgBlurSize % 2 == 0) bgBlurSize += 1;

        cv::Mat denoisedV, bg, diff, shadowThresh;

        // 1. ПОДГОТОВКА ИЗОБРАЖЕНИЯ
        // Используем медианный фильтр с большим ядром 13х13.
        // Он эффективно помогает против гауссовского шума и шума "соль и перец".
        cv::medianBlur(vChannel, denoisedV, 13);

        // Для дополнительной страховки от микро-теней полируем легким Гауссом
        cv::GaussianBlur(denoisedV, denoisedV, cv::Size(3, 3), 0);

        // Размываем УЖЕ ОЧИЩЕННУЮ картинку огромным ядром
        cv::GaussianBlur(denoisedV, bg, cv::Size(bgBlurSize, bgBlurSize), 0);


        // 3. ВЫЧИТАНИЕ (Оставляем только полезную информацию)
        // (из чего вычитаем, что именно, куда сохранить, в пределах какой области)
        cv::subtract(bg, denoisedV, diff, dynamicMask);

        // 4. БИНАРИЗАЦИЯ
        cv::threshold(diff, shadowThresh, shadowContrast, 255, cv::THRESH_BINARY);

        cv::Mat cleanShadow = shadowThresh.clone(); //простое клонирование в новую переменную для удобной отладки

        // ==========================================
        // ШАГ 3: МОРФОЛОГИЯ И ФИЛЬТРЫ
        // ==========================================
        // ЗАКРЫТИЕ, чтобы сначала расширить области (и соединить те участки, что могут потенциально
        // принадлежать тени от гномона, но оказались в разрыве), а затем сузить, чтобы вернуть
        // случайно выделенные шумы к примерно тем размерам, какими они были до Закрытия.
        cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
        cv::morphologyEx(cleanShadow, cleanShadow, cv::MORPH_CLOSE, closeKernel);

        // ОТКРЫТИЕ, чтобы сначала удалить случайно выделенный шум, а потом расширить тень от гномона,
        // потому что операция эрозии вместе с шумом укоротит ещё и тень. Для этого берём ядро размером поменьше,
        // чтобы в начале Открытия не стереть тень совсем.
        cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(cleanShadow, cleanShadow, cv::MORPH_OPEN, openKernel);

        // на готовой картинке ищем контуры теней
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(cleanShadow, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat finalMask = cv::Mat::zeros(frame.size(), CV_8UC1); // снова пустое ч/б изображение

        // импортируем минимальную и максимальную площади контуров из параметров
        double minArea = this->get_parameter("min_contour_area").as_double();
        double maxArea = this->get_parameter("max_contour_area").as_double();
        double maxDistToGnomon = finalRadius * centerTolerance; // максимальное допустимое расстояние от центра гномона до любой точки контура
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
        cv_bridge::CvImage debugMsg(header, sensor_msgs::image_encodings::MONO8, diff);
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
