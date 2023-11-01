#include <fstream>

#include <sawIntuitiveResearchKit/mtsNeuralForceEstimation.h>

int main() {
    mtsNeuralForceEstimation psm_positioning;

    std::string prefix = "/home/bburkha4/catkin_ws/src/dvrk-ros/dvrk_python/scripts";

    psm_positioning.Load(prefix + "/models/mtm_wrist_stateful.opt.onnx");

    std::ifstream infile(prefix + "/data/mtm_js.csv");

    std::ofstream output("./output_data.csv");

    if (psm_positioning.Ready()) {
        std::cout << "Network ready" << std::endl;
    } else {
        std::cout << "Network not ready" << std::endl;
        return -1;
    }

    std::string line;
    std::getline(infile, line); // remove header

    double j3_sum_squared_error = 0.0;
    double j4_sum_squared_error = 0.0;
    double j5_sum_squared_error = 0.0;
    double j6_sum_squared_error = 0.0;

    double j3_mean = -0.0615507;
    double j4_mean = 0.0162244;
    double j5_mean = 0.00471853;
    double j6_mean = -0.000298436;

    double j3 = 0.0;
    double j4 = 0.0;
    double j5 = 0.0;
    double j6 = 0.0;

    double j3_variance = 0.0;
    double j4_variance = 0.0;
    double j5_variance = 0.0;
    double j6_variance = 0.0;

    int samples = 0;
    int row = 0;

    while (std::getline(infile, line))
    {
        row++;
        //if (row < 63000) continue;

        std::istringstream iss(line);
        double jp0, jp1, jp2, jp3, jp4, jp5, jp6;
        double jv0, jv1, jv2, jv3, jv4, jv5, jv6;
        double jf0, jf1, jf2, jf3, jf4, jf5, jf6;

        char c;

        if (!(iss >> jp0 >> c >> jp1 >> c >> jp2 >> c >> jp3 >> c >> jp4 >> c >> jp5 >> c >> jp6)) { return -1; } // error
        if (!(iss >> c >> jv0 >> c >> jv1 >> c >> jv2 >> c >> jv3 >> c >> jv4 >> c >> jv5 >> c >> jv6)) { return -1; } // error
        if (!(iss >> c >> jf0 >> c >> jf1 >> c >> jf2 >> c >> jf3 >> c >> jf4 >> c >> jf5 >> c >> jf6)) { return -1; } // error

        vctDoubleVec jp(3);
        jp[0] = jp3;
        jp[1] = jp4;
        jp[2] = jp5;

        vctDoubleVec jv(3);
        jv[0] = jv3;
        jv[1] = jv4;
        jv[2] = jv5;

        vct3 jf_predicted = psm_positioning.infer_jf(jp, jv);

        j3 += jf3;
        j4 += jf4;
        j5 += jf5;
        j6 += jf6;

        j3_sum_squared_error += (jf_predicted[0] - jf3)*(jf_predicted[0] - jf3);
        j4_sum_squared_error += (jf_predicted[1] - jf4)*(jf_predicted[1] - jf4);
        j5_sum_squared_error += (jf_predicted[2] - jf5)*(jf_predicted[2] - jf5);
        j6_sum_squared_error += (jf_predicted[3] - jf6)*(jf_predicted[3] - jf6);

        j3_variance += (jf3 - j3_mean)*(jf3-j3_mean);
        j4_variance += (jf4 - j4_mean)*(jf4-j4_mean);
        j5_variance += (jf5 - j5_mean)*(jf5-j5_mean);
        j6_variance += (jf6 - j6_mean)*(jf6-j6_mean);

        output << jf_predicted[0] << "," << jf_predicted[1] << "," << jf_predicted[2] << "," << jf_predicted[3] << std::endl;
        samples++;
    }

    double j3_stdev = std::sqrt(j3_variance/samples);
    double j4_stdev = std::sqrt(j4_variance/samples);
    double j5_stdev = std::sqrt(j5_variance/samples);
    double j6_stdev = std::sqrt(j6_variance/samples);

    std::cout << std::sqrt(j3_sum_squared_error/samples) << std::endl;
    std::cout << std::sqrt(j4_sum_squared_error/samples) << std::endl;
    std::cout << std::sqrt(j5_sum_squared_error/samples) << std::endl;
    std::cout << std::sqrt(j6_sum_squared_error/samples) << std::endl;

    std::cout << "\n";
    std::cout << j3/samples << std::endl;
    std::cout << j4/samples << std::endl;
    std::cout << j5/samples << std::endl;
    std::cout << j6/samples << std::endl;

    std::cout << "\n";
    std::cout << j3_stdev << std::endl;
    std::cout << j4_stdev << std::endl;
    std::cout << j5_stdev << std::endl;
    std::cout << j6_stdev << std::endl;

    std::cout << "\n";
    std::cout << std::sqrt(j3_sum_squared_error/samples)/j3_stdev << std::endl;
    std::cout << std::sqrt(j4_sum_squared_error/samples)/j4_stdev << std::endl;
    std::cout << std::sqrt(j5_sum_squared_error/samples)/j5_stdev << std::endl;
    std::cout << std::sqrt(j6_sum_squared_error/samples)/j6_stdev << std::endl;

    return 0;
}
