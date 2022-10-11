// Copyright (C) 2021-2022 Petter Nilsson. MIT License.

#pragma once

#include <Eigen/Core>

inline auto Misra1a()
{
  auto f = [](double y, double x, const Eigen::VectorXd & p) -> double { return y - (p(0) * (1 - exp(-p(1) * x))); };

  Eigen::MatrixXd data(14, 2);
  Eigen::VectorXd start1(2), start2(2), optim(2);

  // clang-format off
  data <<
        10.07E0,      77.6E0,
        14.73E0,     114.9E0,
        17.94E0,     141.1E0,
        23.93E0,     190.8E0,
        29.61E0,     239.9E0,
        35.18E0,     289.0E0,
        40.02E0,     332.8E0,
        44.82E0,     378.4E0,
        50.76E0,     434.8E0,
        55.05E0,     477.3E0,
        61.01E0,     536.8E0,
        66.40E0,     593.1E0,
        75.47E0,     689.1E0,
        81.78E0,     760.0E0;

  start1 << 500, 0.0001;
  start2 << 250, 0.0005;
  optim <<
      2.3894212918E+02,
      5.5015643181E-04;
  // clang-format on

  return std::make_tuple(std::move(f), std::move(data), std::move(start1), std::move(start2), std::move(optim));
}

inline auto Kirby2()
{
  auto f = [](double y, double x, const Eigen::VectorXd & p) -> double {
    return y - (p(0) + p(1) * x + p(2) * x * x) / (1. + p(3) * x + p(4) * x * x);
  };

  Eigen::VectorXd start1(5), start2(5), optim(5);
  Eigen::MatrixXd data(151, 2);

  // clang-format off
  data <<
       0.0082E0,      9.65E0,
       0.0112E0,     10.74E0,
       0.0149E0,     11.81E0,
       0.0198E0,     12.88E0,
       0.0248E0,     14.06E0,
       0.0324E0,     15.28E0,
       0.0420E0,     16.63E0,
       0.0549E0,     18.19E0,
       0.0719E0,     19.88E0,
       0.0963E0,     21.84E0,
       0.1291E0,     24.00E0,
       0.1710E0,     26.25E0,
       0.2314E0,     28.86E0,
       0.3227E0,     31.85E0,
       0.4809E0,     35.79E0,
       0.7084E0,     40.18E0,
       1.0220E0,     44.74E0,
       1.4580E0,     49.53E0,
       1.9520E0,     53.94E0,
       2.5410E0,     58.29E0,
       3.2230E0,     62.63E0,
       3.9990E0,     67.03E0,
       4.8520E0,     71.25E0,
       5.7320E0,     75.22E0,
       6.7270E0,     79.33E0,
       7.8350E0,     83.56E0,
       9.0250E0,     87.75E0,
      10.2670E0,     91.93E0,
      11.5780E0,     96.10E0,
      12.9440E0,    100.28E0,
      14.3770E0,    104.46E0,
      15.8560E0,    108.66E0,
      17.3310E0,    112.71E0,
      18.8850E0,    116.88E0,
      20.5750E0,    121.33E0,
      22.3200E0,    125.79E0,
      22.3030E0,    125.79E0,
      23.4600E0,    128.74E0,
      24.0600E0,    130.27E0,
      25.2720E0,    133.33E0,
      25.8530E0,    134.79E0,
      27.1100E0,    137.93E0,
      27.6580E0,    139.33E0,
      28.9240E0,    142.46E0,
      29.5110E0,    143.90E0,
      30.7100E0,    146.91E0,
      31.3500E0,    148.51E0,
      32.5200E0,    151.41E0,
      33.2300E0,    153.17E0,
      34.3300E0,    155.97E0,
      35.0600E0,    157.76E0,
      36.1700E0,    160.56E0,
      36.8400E0,    162.30E0,
      38.0100E0,    165.21E0,
      38.6700E0,    166.90E0,
      39.8700E0,    169.92E0,
      40.0300E0,    170.32E0,
      40.5000E0,    171.54E0,
      41.3700E0,    173.79E0,
      41.6700E0,    174.57E0,
      42.3100E0,    176.25E0,
      42.7300E0,    177.34E0,
      43.4600E0,    179.19E0,
      44.1400E0,    181.02E0,
      44.5500E0,    182.08E0,
      45.2200E0,    183.88E0,
      45.9200E0,    185.75E0,
      46.3000E0,    186.80E0,
      47.0000E0,    188.63E0,
      47.6800E0,    190.45E0,
      48.0600E0,    191.48E0,
      48.7400E0,    193.35E0,
      49.4100E0,    195.22E0,
      49.7600E0,    196.23E0,
      50.4300E0,    198.05E0,
      51.1100E0,    199.97E0,
      51.5000E0,    201.06E0,
      52.1200E0,    202.83E0,
      52.7600E0,    204.69E0,
      53.1800E0,    205.86E0,
      53.7800E0,    207.58E0,
      54.4600E0,    209.50E0,
      54.8300E0,    210.65E0,
      55.4000E0,    212.33E0,
      56.4300E0,    215.43E0,
      57.0300E0,    217.16E0,
      58.0000E0,    220.21E0,
      58.6100E0,    221.98E0,
      59.5800E0,    225.06E0,
      60.1100E0,    226.79E0,
      61.1000E0,    229.92E0,
      61.6500E0,    231.69E0,
      62.5900E0,    234.77E0,
      63.1200E0,    236.60E0,
      64.0300E0,    239.63E0,
      64.6200E0,    241.50E0,
      65.4900E0,    244.48E0,
      66.0300E0,    246.40E0,
      66.8900E0,    249.35E0,
      67.4200E0,    251.32E0,
      68.2300E0,    254.22E0,
      68.7700E0,    256.24E0,
      69.5900E0,    259.11E0,
      70.1100E0,    261.18E0,
      70.8600E0,    264.02E0,
      71.4300E0,    266.13E0,
      72.1600E0,    268.94E0,
      72.7000E0,    271.09E0,
      73.4000E0,    273.87E0,
      73.9300E0,    276.08E0,
      74.6000E0,    278.83E0,
      75.1600E0,    281.08E0,
      75.8200E0,    283.81E0,
      76.3400E0,    286.11E0,
      76.9800E0,    288.81E0,
      77.4800E0,    291.08E0,
      78.0800E0,    293.75E0,
      78.6000E0,    295.99E0,
      79.1700E0,    298.64E0,
      79.6200E0,    300.84E0,
      79.8800E0,    302.02E0,
      80.1900E0,    303.48E0,
      80.6600E0,    305.65E0,
      81.2200E0,    308.27E0,
      81.6600E0,    310.41E0,
      82.1600E0,    313.01E0,
      82.5900E0,    315.12E0,
      83.1400E0,    317.71E0,
      83.5000E0,    319.79E0,
      84.0000E0,    322.36E0,
      84.4000E0,    324.42E0,
      84.8900E0,    326.98E0,
      85.2600E0,    329.01E0,
      85.7400E0,    331.56E0,
      86.0700E0,    333.56E0,
      86.5400E0,    336.10E0,
      86.8900E0,    338.08E0,
      87.3200E0,    340.60E0,
      87.6500E0,    342.57E0,
      88.1000E0,    345.08E0,
      88.4300E0,    347.02E0,
      88.8300E0,    349.52E0,
      89.1200E0,    351.44E0,
      89.5400E0,    353.93E0,
      89.8500E0,    355.83E0,
      90.2500E0,    358.32E0,
      90.5500E0,    360.20E0,
      90.9300E0,    362.67E0,
      91.2000E0,    364.53E0,
      91.5500E0,    367.00E0,
      92.2000E0,    371.30E0;

  start1 <<
    2,
    -0.1,
    0.003,
    -0.001,
    0.00001;

  start2 <<
    1.5,
    -0.15,
    0.0025,
    -0.0015,
    0.00002;

  optim <<
    1.6745063063E+00,
    -1.3927397867E-01,
    2.5961181191E-03,
    -1.7241811870E-03,
    2.1664802578E-05;
  // clang-format on

  return std::make_tuple(std::move(f), data, start1, start2, optim);
}

inline auto MGH09()
{
  auto f = [](double y, double x, const Eigen::VectorXd & p) -> double {
    return y - p(0) * (x * x + x * p(1)) / (x * x + x * p(2) + p(3));
  };

  Eigen::MatrixXd data(11, 2);
  Eigen::VectorXd start1(4), start2(4), optim(4);

  // clang-format off
  data <<
       1.957000E-01,    4.000000E+00,
       1.947000E-01,    2.000000E+00,
       1.735000E-01,    1.000000E+00,
       1.600000E-01,    5.000000E-01,
       8.440000E-02,    2.500000E-01,
       6.270000E-02,    1.670000E-01,
       4.560000E-02,    1.250000E-01,
       3.420000E-02,    1.000000E-01,
       3.230000E-02,    8.330000E-02,
       2.350000E-02,    7.140000E-02,
       2.460000E-02,    6.250000E-02;

  start1 <<
    25,
    39,
    41.5,
    39;

  start2 <<
    0.25,
    0.39,
    0.415,
    0.39;

  optim <<
    1.9280693458E-01,
    1.9128232873E-01,
    1.2305650693E-01,
    1.3606233068E-01;
  // clang-format on

  return std::make_tuple(std::move(f), data, start1, start2, optim);
}
