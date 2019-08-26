#include <iostream>
#include <cmath>
#include <vector>

#include "fftw3.h"

double gauss(double x, double m, double s)
{
  double t1 = 1. / (sqrt(2 * M_PI) * s);
  double t2 = exp( -((x - m) * (x - m)) / (2 * s * s) );

  return t1 * t2;
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

  int mid = static_cast<int>(floor(n / 2));

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
  std::vector<double> filter = makeGauss(0, 3, -30, 30, 100);

  std::vector<double> conv = convolve(data, filter);

  for (auto v : conv){
    std::cout << v << std::endl;
  }

  return 0;
}
