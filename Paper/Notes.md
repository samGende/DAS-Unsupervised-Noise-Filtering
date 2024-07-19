# Runtime Notes

### Runtime with loop for time cwt and loop for space cwt

filter took 1.132530927658081

time cwt took 18.942780017852783

space cwt took 9.199059247970581

filter took 1.3635871410369873

time cwt took 20.480812072753906

space cwt took 8.80389404296875

### Runtime with vectorized time cwt and loop for space cwt

filter took 1.180962085723877

invers fft took 19.92713189125061

time cwt took 25.332064867019653

space cwt took 10.487303972244263

filter took 0.978571891784668

invers fft took 26.3227481842041

time cwt took 31.879231929779053

space cwt took 8.40162181854248

### Runtime for vectorized cwt time and space cwt

filter took 1.1873281002044678

invers fft took 16.410244941711426

space cwt took 26.059340000152588

invers fft took 26.68893003463745

time cwt took 39.94915819168091

filter took 1.003904104232788

invers fft took 5.566688060760498

space cwt took 11.63805603981018

invers fft took 18.75801706314087

time cwt took 28.544529914855957

### Colab GPU runtime with vectorized cwt time and space

filter took 0.5003564357757568

multiplication took -0.8215184211730957

invers fft took 5.379493713378906

space cwt took 9.06934142112732

invers fft took 14.798910140991211

time cwt took 16.988465785980225
filter took 0.47977471351623535

multiplication took -0.8040425777435303

invers fft took 6.5471484661102295

space cwt took 10.39218020439148

invers fft took 12.29993748664856

time cwt took 14.5299813747406

### Colab GPU runtime fully vectorized with cupy instead of numpy

filter took 0.4657771587371826
signal shape inside of space vec (14999, 286)
multiplication took -0.0040357112884521484
shape after multiplication in space (30, 14999, 286)
invers fft took 0.05071091651916504
space cwt took 0.061079978942871094
(30, 14999, 286)

### formula to calculate sacles from frequency where $ \lambda$ = frequency

$$ scale = \frac{\lambda(\omega_0 + \sqrt{2 + \omega_0})}{4\pi} $$

### formula for angular frequencies

$$
w_k = \begin{cases}
   \frac{2 \pi k}{dt} & \text{ k $\le \frac{N}{2} $ }\\
   -\frac{2 \pi k}{dt} & \text{ k >$\ \frac{N}{2} $ }\\
    \end{cases}
$$

### formula for morlet wavelet

$$
\psi_0(N) = \pi^{ -\frac{1}{4}} e^{-iw_0N} e^{-n^2/2}
$$

### formula for morlet wavelet at a scale

$$
\psi_0(sw) = \pi^{ -\frac{1}{4}} H(w) e^{-(sw - w_0)^2/2}
$$

### Normalization term for wavelet

$$ \sqrt{ \frac{2\pi s}{dt}}$$

### error with agglomorative clustering

numpy.core.\_exceptions.MemoryError: Unable to allocate 268. TiB for an array with shape (36808195710000,) and data type float64
