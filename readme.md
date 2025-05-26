# Traffic Audio Monitoring Using Neural Networks on Microcontrollers

A thesis project evaluating the feasibility of using Convolutional Neural Networks (CNNs) to classify and count vehicles based on audio input, all running on cost-efficient microcontrollers.

![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)
![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Raspberry](https://img.shields.io/badge/Raspberry%20Pi-A22846?style=for-the-badge&logo=Raspberry%20Pi&logoColor=white)
![LaTex](https://img.shields.io/badge/LaTeX-47A141?style=for-the-badge&logo=LaTeX&logoColor=white)

## 📘 Abstract

This study evaluates the feasibility of classifying and counting bypassing
vehicles based on audio using Convolutional Neural Networks (CNNs)
on a cost-efficient Microcontroller Units (MCUs). The classification task
involved distinguishing four classes: car, motorcycle, commercial vehicle, and
background noise.

A lightweight CNN was trained on extracted Mel-frequency cepstrum
coefficients (MFCCs) (audio features) from a pre-recorded dataset. Software-
based tests were conducted to see if the model could perform well without the
MCU, while lab-based tests were done after the model was deployed on the
MCU. The software-based test used TensorFlow to evaluate the classification
rate. In contrast, the lab-based test used a program built to simulate an audio
stream to the MCU alongside a classification voting process to compensate for
short audio frames.

The results showed a classification accuracy of 84.8% and F1-score of
87.0% during the software-based test. While the lab-based test showed a
higher classification accuracy of 88.4% and F1-score of 90.4%. The lab-based
test also showed a vehicle counting accuracy of 99.8%.

The study’s goal was to investigate the feasibility of using CNNs on-
board MCUs for classifying and counting passing vehicles. By following
the guidelines in the report, the results confirm that this is indeed a feasible
alternative to other vehicle counting alternatives. It also serves as a foundation
for using classification in real time.


## 📊 Results

| Test Type       | Accuracy | F1-Score | Counting Accuracy |
|----------------|----------|----------|-------------------|
| Software-based | 84.8%    | 87.0%    | -                 |
| Lab-based MCU  | 88.4%    | 90.4%    | 99.8%             |

## 📦 Dataset

- **Name**: IDMT-traffic
- **Classes**: `car`, `motorcycle`, `commercial vehicle` (bus/truck), `background_noise`
- **Source**: Jakob Abeßer, Saichand Gourishetti, András Kátai, Tobias Clauß, Prachi Sharma, Judith Liebetrau IDMT-Traffic: An Open Benchmark Dataset for Acoustic Traffic Monitoring Research, EUSIPCO, 2021.
 

## ⚙️ Microcontroller Specs

- **Device**: Raspberry Pi Pico 2W
- **Constraints**: Memory and compute optimized
- **Implementation**: Audio framing (0.25s), soft plurality voting for temporal accuracy

## 👥 Authors

- [William Frid](https://github.com/williamfridh)
- [Pontus Åhlin](https://github.com/PontusAhlin)

> Bachelor's Programme in Information and Communication Technology  
> School of Electrical Engineering and Computer Science, KTH Royal Institute of Technology  