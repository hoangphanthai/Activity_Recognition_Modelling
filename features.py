import numpy as np
import math
from scipy.stats import kurtosis, skew, entropy

def centervalue(x):
    y = np.array(x)
    center = np.take(y, y.size // 2)
    # center = np.mean(y)
    return (len(y)*float("{0:.3f}".format(center)))/len(y)

# Time domain features =>

def IQR(x):
    interqtlrange = np.percentile(x, 75) - np.percentile(x, 25)
    return (len(x)*float("{0:.3f}".format(interqtlrange)))/len(x)

def RMS(x): # Calculate Root Mean Square
    return float("{0:.3f}".format(math.sqrt(sum(n*n for n in x)/len(x))))

def MCR(x): # Calculate the Mean Crossing Rate
    my_array = np.array(x) - np.mean(x)
    sum = int(0)
    for i in range(len(my_array)-1):
        if my_array[i]*my_array[i+1]<0:
            sum = sum + 1
    return (sum/len(x))

def Kurt(x): # Calculate the Kurtosis
    return (len(x)*float("{0:.3f}".format(kurtosis(x))))/len(x)
    # return float("{0:.3f}".format(kurtosis(x)))

def Skew(x): # Calculate the Skewness
    return (len(x)*float("{0:.3f}".format(skew(x))))/len(x)
# End domain features <=

# Frequency domain features =>
def frEnergy(x): # Spectral Energy of the signal
    # return np.sum(x**2)/len(x)
    n = len(x)
    mn = np.mean(x)
    # Remove DC Component
    xf = np.array(x) - mn
    # FFT Spectrum
    fft_spectrum = np.fft.rfft(xf)
    # FFT Normalized Power Spectrum
    energy = np.sum(np.abs(fft_spectrum))/n
    return energy

def frPeakFreq(x, sample_rate): # Frequency component that has the greatest magnitude
    n = len(x)
    mn = np.mean(x)
    # Remove DC Component
    xf = np.array(x) - mn
    # FFT Spectrum
    fft_spectrum = np.fft.rfft(xf)
    # FFT Scaled Power Spectrum
    mag = np.abs(fft_spectrum)/n
    # Frequencies
    f = np.linspace(0, sample_rate/2, len(mag))  # 10 Samples/sec, so 5 Hz is Nyquist limit
    # print('Principal frequency - the highest peak')
    peak_idx = np.argpartition(mag, -1)[-1:]
    # print(f[peak_idx][0])
    return f[peak_idx][0]

def frDmEntroPy(x): #Frequency domain entropy, also known as a (Power) Spectral Entropy
    np.seterr(divide='ignore', invalid='ignore')
    n = len(x)
    mn = np.mean(x)
    # Remove DC Component
    xf = np.array(x) - mn
    # FFT discrete Fourier Transform
    fft_spectrum = np.fft.rfft(xf)
    # FFT Scaled Power Spectrum
    mag = np.abs(fft_spectrum)/n
    # Normalize the Power Spectrum
    pi = mag/np.sum(mag)
    # Calculate the Power Spectral Entropy
    pse = entropy(pi)
    return pse

def frMag1(x): # Magnitude of first component of FFT analysis
    n = len(x)
    mn = np.mean(x)
    # Remove DC Component
    xf = np.array(x) - mn
    # FFT Spectrum
    fft_spectrum = np.fft.rfft(xf)
    # FFT Scaled Power Spectrum
    mag = np.abs(fft_spectrum)/n
    if len(mag)>=2:
        return mag[1]
    else:
        return 0

def frMag2(x): # Magnitude of second component of FFT analysis
    n = len(x)
    mn = np.mean(x)
    # Remove DC Component
    xf = np.array(x) - mn
    # FFT Spectrum
    fft_spectrum = np.fft.rfft(xf)
    # FFT Scaled Power Spectrum
    mag = np.abs(fft_spectrum)/n
    if len(mag) >= 3:
        return mag[2]
    else:
        return 0

def frMag3(x): # Magnitude of third component of FFT analysis
    n = len(x)
    mn = np.mean(x)
    # Remove DC Component
    xf = np.array(x) - mn
    # FFT Spectrum
    fft_spectrum = np.fft.rfft(xf)
    # FFT Scaled Power Spectrum
    mag = np.abs(fft_spectrum)/n
    if len(mag) >= 4:
        return mag[3]
    else:
        return 0

def frMag4(x): # Magnitude of fourth component of FFT analysis
    n = len(x)
    mn = np.mean(x)
    # Remove DC Component
    xf = np.array(x) - mn
    # FFT Spectrum
    fft_spectrum = np.fft.rfft(xf)
    # FFT Scaled Power Spectrum
    mag = np.abs(fft_spectrum)/n
    if len(mag) >= 5:
        return mag[4]
    else:
        return 0

def frMag5(x): # Magnitude of fifth component of FFT analysis
    n = len(x)
    mn = np.mean(x)
    # Remove DC Component
    xf = np.array(x) - mn
    # FFT Spectrum
    fft_spectrum = np.fft.rfft(xf)
    # FFT Scaled Power Spectrum
    mag = np.abs(fft_spectrum)/n
    if len(mag) >= 6:
        return mag[5]
    else:
        return 0
# Frequency domain features <=