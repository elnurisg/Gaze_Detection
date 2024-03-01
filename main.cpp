#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;
using namespace std;


// ----- PARAMETERS -----
int START_R = 2;
int END_R = 200;
int STEP = 1;
int WHITE_NEIGHBOR_R = 20;
float WHITE_DIFF_RATIO = 0.2;
float WHITE_VOTE_WEIGHT = 0;
// ----------------------


int PRED_LEFT = 0;
int PRED_RIGHT = 1;
int PRED_CENTER = 2;

CascadeClassifier eyes_cascade;

/*
    Cut the eyes out of the frame
*/
std::vector<Mat> cut_eyes(Mat frame)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    // equalizeHist(frame_gray, frame_gray);

    std::vector<Rect> eyes;
    eyes_cascade.detectMultiScale(frame_gray, eyes);

    std::vector<Mat> eyes_cropped;
    for (size_t i = 0; i < eyes.size(); i++)
    {
        eyes_cropped.push_back(frame(eyes[i]));
    }

    return eyes_cropped;
}

/*
    Detect the pupil from eye image
*/
Vec3f detect_pupil_point(Mat eye_gray, int center_x, int center_y)
{

    int width = eye_gray.cols;
    int height = eye_gray.rows;

    int start_r = START_R;
    int end_r = END_R;

    int step = STEP;

    int x = center_x;
    int y = center_y;

    std::vector<float> intensities;
    Mat mask = Mat::zeros(eye_gray.size(), CV_8UC1);

    std::vector<int> radii;
    for (int r = start_r; r < end_r; r += step)
    {
        circle(mask, Point(x, y), r, Scalar(255), 1, LINE_AA);
        Mat diff = eye_gray & mask;
        intensities.push_back(sum(diff)[0] / (CV_PI * r * r));
        radii.push_back(r);
        mask = Mat::zeros(eye_gray.size(), CV_8UC1);
    }

    std::vector<float> intensities_np(intensities.begin(), intensities.end());
    intensities.clear();

    for (int i = 0; i < intensities_np.size() - 1; i++)
    {
        intensities_np[i] = intensities_np[i] - intensities_np[i + 1];
    }

    Mat kernel = (Mat_<float>(1, 3) << 0.1068, 0.7866, 0.1068);
    filter2D(intensities_np, intensities_np, -1, kernel);

    int idx = 0;
    float max = intensities_np[0];
    for (int i = 1; i < intensities_np.size(); i++)
    {
        if (intensities_np[i] > max)
        {
            max = intensities_np[i];
            idx = i;
        }
    }

    return Vec3f(x, y, radii[idx]);
}

Vec3f detect_pupil(Mat eye)
{
    // calls detect_pupil_point for every point in the center third of the image
    // and returns the point with the highest intensity

    int width = eye.cols;
    int height = eye.rows;

    int center_x = width / 2;
    int center_y = height / 2;

    int start_x = center_x - width / 3;
    int end_x = center_x + width / 3;

    int start_y = center_y - height / 3;
    int end_y = center_y + height / 3;

    Vec3f best_pupil = Vec3f(0, 0, 0);
    float best_intensity = 0;

    Mat eye_gray;
    cvtColor(eye, eye_gray, COLOR_BGR2GRAY);
    equalizeHist(eye_gray, eye_gray);

    for (int x = start_x; x < end_x; x++)
    {
        for (int y = start_y; y < end_y; y++)
        {
            Vec3f pupil = detect_pupil_point(eye_gray, x, y);
            float intensity = pupil[2];
            if (intensity > best_intensity)
            {
                best_intensity = intensity;
                best_pupil = pupil;
            }
        }
    }

    return best_pupil;
}


/*
    Weak classifier to predict the gaze direction based on intensity
    of the WHITE_NEIGHBOR_R neighboring pixels along the horizontal line
    through the pupil. If the left side is brighter, the gaze is predicted
    to be left, if the right side is brighter, the gaze is predicted to be
    right, otherwise the gaze is predicted to be center.
*/
int predict_white(Mat eye, Vec3f pupil)
{
    int x = pupil[0];
    int y = pupil[1];

    int start_x = x - WHITE_NEIGHBOR_R;
    int end_x = x + WHITE_NEIGHBOR_R;

    // make sure the start and end points are within the image
    if (start_x < 0)
    {
        start_x = 0;
    }
    if (end_x > eye.cols)
    {
        end_x = eye.cols;
    }

    Mat eye_gray;
    cvtColor(eye, eye_gray, COLOR_BGR2GRAY);
    // equalizeHist(eye_gray, eye_gray);

    int left = 0;
    int right = 0;

    for (int i = start_x; i < end_x; i++)
    {
        if (i < x)
        {
            left += eye_gray.at<uchar>(y, i);
        }
        else
        {
            right += eye_gray.at<uchar>(y, i);
        }
    }

    // decide based on ratio of the intensity difference
    float diff = left - right;
    float ratio = diff / (left + right);
    if (ratio > WHITE_DIFF_RATIO)
    {
        return PRED_LEFT;
    }
    else if (ratio < -WHITE_DIFF_RATIO)
    {
        return PRED_RIGHT;
    }
    else
    {
        return PRED_CENTER;
    }
}


/*
    Detect the gaze direction
*/
int detect(Mat frame)
{
    std::vector<Mat> eyes = cut_eyes(frame);

    if (eyes.size() == 0)
    {
        cout << "No eyes detected" << endl;
        return PRED_CENTER;
    }
    // cout << "Eyes detected: " << eyes.size() << endl;

    // detect the gaze direction based on the pupil position
    // detect pupil for all found eyes and return the most likely direction
    // if there are no eyes found, return PRED_CENTER
    
    // int i = 0;
    // for(Mat eye : eyes){
        // Vec3f pupil = detect_pupil(eye);
        // cout << pupil << endl;
        // circle(eye, Point(pupil[0], pupil[1]), pupil[2], Scalar(0, 0, 255), 1, LINE_AA);
        // imshow(std::to_string(i), eye);
        // i++;
    // }
    // waitKey(0);

    float left = 0;
    float right = 0;
    float center = 0;

    for (Mat eye : eyes)
    {
        Vec3f pupil = detect_pupil(eye);

        if (pupil[0] < eye.cols / 2)
        {
            right++;
        }
        else if (pupil[0] > eye.cols / 2)
        {
            left++;
        }
        else
        {
            center++;
        }

        // add the prediction based on the white pixels to the vote
        int direction_from_white = predict_white(eye, pupil);
        // cout << "Direction from white: " << direction_from_white << endl;
        if (direction_from_white == PRED_LEFT)
        {
            left += WHITE_VOTE_WEIGHT;
        }
        else if (direction_from_white == PRED_RIGHT)
        {
            right += WHITE_VOTE_WEIGHT;
        }
        else
        {
            center += WHITE_VOTE_WEIGHT;
        }
    }

    if (left > right && left > center) return PRED_LEFT;
    else if (right > left && right > center) return PRED_RIGHT;
    else {
        return PRED_CENTER;
    }
}

/*
    Evaluate the algorithm on a test set.
    The test set is in the folder data and organized by the label,
    such that /data/left contains images with the label left and etc.
    This function computes the confusion matrix and the accuracy.
*/
std::pair<float, Mat> evaluate()
{
    std::vector<std::string> labels = {
        "left", "right", "center"};
    std::vector<std::string> images = {
        "left01.jpg", "left02.jpg", "left03.jpg", "left04.jpg", "left05.png", "left06.jpg", "left07.jpg", "left08.jpg", "left09.jpg", "left10.png",
        "right01.png", "right02.jpg", "right03.jpg", "right04.jpg", "right05.jpg", "right06.jpg", "right07.jpg", "right08.jpg", "right09.jpg", "right10.jpg",
        "center01.png", "center02.jpg", "center03.jpg", "center04.jpg", "center05.jpg", "center06.jpg", "center07.jpeg", "center08.jpg", "center09.jpg", "center10.jpg"};

    Mat confusion_matrix = Mat::zeros(3, 3, CV_32F);

    for (int i = 0; i < images.size(); i++)
    {
        int label_idx = i / 10;
        std::string image_path = "data/" + labels[label_idx] + "/" + images[i];
        cout << "Processing image " << image_path << endl;
        Mat img = imread(image_path);

        if (img.empty())
        {
            cout << "Could not read the image: " << image_path << endl;
            return std::pair<float, Mat>(0, confusion_matrix);
        }

        int pred = detect(img);
        cout << "Direction: " << pred << endl;
        confusion_matrix.at<float>(label_idx, pred) += 1;
    }

    float accuracy = 0;
    for (int i = 0; i < 3; i++)
    {
        accuracy += confusion_matrix.at<float>(i, i);
    }
    accuracy /= images.size();

    return std::pair<float, Mat>(accuracy, confusion_matrix);
}

int main()
{

    // std::string image_path = "data/right/right05.jpg";
    // Mat img = imread(image_path, IMREAD_COLOR);

    // if (img.empty())
    // {
        // cout << "Could not read the image: " << image_path << endl;
        // return 1;
    // }

    eyes_cascade.load("haarcascade_eye_tree_eyeglasses.xml");

    // detect(img);

    std::pair<float, Mat> result = evaluate();
    cout << "------------------------" << endl;
    cout << "Evaluation results:" << endl;
    cout << "Accuracy: " << result.first << endl;
    cout << "Confusion matrix: (L R C)" << endl
         << result.second << endl;

        return 0;
}