#include "lm_solver_data.h"

constexpr std::array x_data_3 {
    1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
    13.0, 14.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
    26.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 37.0, 38.0, 39.0,
    40.0, 41.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0,
    53.0, 54.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0,
    66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0,
    78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0,
    90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0
};

constexpr std::array y_data_3 { 228.571428571429,
                                234.285714285714,
                                205.714285714286,
                                168.571428571429,
                                190.0,
                                164.285714285714,
                                171.428571428571,
                                192.857142857143,
                                144.285714285714,
                                142.857142857143,
                                170.0,
                                115.714285714286,
                                131.428571428571,
                                128.571428571429,
                                127.142857142857,
                                135.714285714286,
                                111.428571428571,
                                115.714285714286,
                                118.571428571429,
                                107.142857142857,
                                108.571428571429,
                                81.4285714285714,
                                81.4285714285714,
                                78.5714285714286,
                                58.5714285714286,
                                67.1428571428571,
                                84.2857142857143,
                                57.1428571428571,
                                68.5714285714286,
                                50.0,
                                71.4285714285714,
                                57.1428571428571,
                                50.0,
                                57.1428571428571,
                                51.4285714285714,
                                50.0,
                                55.7142857142857,
                                55.7142857142857,
                                48.5714285714286,
                                42.8571428571429,
                                42.8571428571429,
                                37.1428571428571,
                                31.4285714285714,
                                37.1428571428571,
                                28.5714285714286,
                                40.0,
                                32.8571428571429,
                                38.5714285714286,
                                41.4285714285714,
                                34.2857142857143,
                                25.7142857142857,
                                32.8571428571429,
                                37.1428571428571,
                                22.8571428571429,
                                34.2857142857143,
                                22.8571428571429,
                                37.1428571428571,
                                28.5714285714286,
                                22.8571428571429,
                                28.5714285714286,
                                27.1428571428571,
                                32.8571428571429,
                                32.8571428571429,
                                25.7142857142857,
                                22.8571428571429,
                                21.4285714285714,
                                21.4285714285714,
                                24.2857142857143,
                                17.1428571428571,
                                28.5714285714286,
                                25.7142857142857,
                                22.8571428571429,
                                25.7142857142857,
                                20.0,
                                28.5714285714286,
                                18.5714285714286,
                                25.7142857142857,
                                25.7142857142857,
                                24.2857142857143,
                                27.1428571428571,
                                25.7142857142857,
                                25.7142857142857,
                                20.0,
                                17.1428571428571,
                                22.8571428571429,
                                25.7142857142857,
                                20.0,
                                18.5714285714286,
                                21.4285714285714,
                                21.4285714285714,
                                20.0,
                                17.1428571428571,
                                12.8571428571429,
                                22.8571428571429,
                                17.1428571428571 };
