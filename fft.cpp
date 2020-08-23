#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#include "fftw3.h"

double gauss(double x, double m, double s)
{
  double t1 = 1. / (sqrt(2 * M_PI) * s);
  double t2 = exp( -((x - m) * (x - m)) / (2 * s * s) );

  return t1 * t2;
}

double gauss2d(double x1, double x2, double m1, double m2, double s1, double s2, double rho)
{
    double z1 = ((x1 - m1) * (x1 - m1)) / (s1 * s1);
    double z2 = ((x2 - m2) * (x2 - m2)) / (s2 * s2);
    double zCross = (2 * rho * (x1 - m1) * (x2 - m2)) / (s1 * s2);

    double z = z1 + z2 + zCross;

    double norm = 1 / (2 * M_PI * s1 * s2 * sqrt(1 - rho * rho));

    double arg = -z / (2 * (1 - rho * rho));

    return norm * exp(arg);
}

std::vector<double> makeGauss(double m, double s, double low, double high, int n)
{
  std::vector<double> out(n);

  double step_size = (high - low) / n;

  for (int i = 0; i < n; i++) {
    double x = low + step_size * i;
    out[i] = gauss(x, m, s);
  }

  return out;

}

std::vector<std::vector<double>> makeGauss2d(double m1, double m2, double s1, double s2, double rho, double low, double high, int n)
{
  std::vector<std::vector<double>> out(n, std::vector<double>(n));

  double step_size = (high - low) / n;

  for (int i = 0; i < n; i++) {

    double x1 = low + step_size * i;

    for (int j = 0; i < n; j++) {

      double x2 = low + step_size * j;

      out[i][j] = gauss2d(x1, x2, m1, m2, s1, s2, rho);
    }
  }

  return out;

}

fftw_complex * computeFFT(fftw_complex * in, int n, int sign)
{

  fftw_complex * out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * n);

  fftw_plan plan = fftw_plan_dft_1d(n, in, out, sign, FFTW_ESTIMATE);

  fftw_execute(plan);

  fftw_destroy_plan(plan);

  return out;
}

fftw_complex * packVectorReal(std::vector<double> * in)
{
  int n = in->size();
  fftw_complex * out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * n);

  for (int i = 0; i < n; i++){
    out[i][0] = in->at(i);
    out[i][1] = 0.; // Imag == 0
  }

  return out;

}

fftw_complex * multiply(fftw_complex * a, fftw_complex * b, int n)
{
  fftw_complex * out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * n);

  for (int i = 0; i < n; i++){
    out[i][0] = a[i][0] * b[i][0];
    out[i][1] = a[i][1] * b[i][1];
  }

  return out;
}

std::vector<double> convolve(std::vector<double> & a, std::vector<double> & b)
{
  int n = a.size();

  fftw_complex * a_complex = packVectorReal(&a);
  fftw_complex * b_complex = packVectorReal(&b);

  fftw_complex * a_fft = computeFFT(a_complex, n, FFTW_FORWARD);
  fftw_complex * b_fft = computeFFT(b_complex, n, FFTW_FORWARD);

  fftw_complex * to_conv = multiply(a_fft, b_fft, n);

  fftw_complex * conv = computeFFT(to_conv, n, FFTW_BACKWARD);

  std::vector<double> conv_real(n);

  int mid = (n % 2 != 0) ? (n / 2) + 1 : (n / 2);

  for(int i = mid; i < n; i++){
    conv_real[i - mid] = conv[i][0];
  }

  for(int i = 0; i < mid; i++){
    conv_real[mid + i] = conv[i][0];
  }

  fftw_free(a_complex);
  fftw_free(b_complex);

  fftw_free(a_fft);
  fftw_free(b_fft);

  fftw_free(to_conv);
  fftw_free(conv);

  return conv_real;
}

int main(int argc, char const *argv[])
{

  std::vector<double> data = makeGauss(0, 3, -30, 30, 100);
  std::vector<double> filter = makeGauss(0, 5, -30, 30, 100);

  std::vector<double> conv = convolve(data, filter);

  std::ofstream fileOut;

  fileOut.open ("test.csv");

  for (auto v : conv){
    fileOut << v << std::endl;
  }

  fileOut.close();

  return 0;
}
