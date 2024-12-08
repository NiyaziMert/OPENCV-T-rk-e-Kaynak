# OPENCV-Turkce-Kaynak
# 📘 OpenCV ve Python ile Görüntü İşleme Rehberi

Bu rehber, **OpenCV ve Python** ile **görüntü işleme** konusundaki temel ve ileri düzey kavramları, kod örnekleriyle açıklamaktadır. Bu kaynak, hem yeni başlayanlar hem de ileri düzey kullanıcılar için yararlı olacaktır. Rehber, **görüntü filtreleme, köşe tespiti, morfolojik işlemler, obje tespiti** gibi konularda kapsamlı bir kılavuz sunar.

---

## 📢 **Önsöz**
Bu doküman, **Niyazi Mert Işıksal** tarafından hazırlanmış kapsamlı bir **OpenCV ve Python Görüntü İşleme Kılavuzu**dur. İçerdiği bilgiler, **görüntü işlemede en sık kullanılan yöntemleri ve algoritmaları** öğretmeyi amaçlamaktadır. 

Bu kılavuzun amacı, **görüntü işleme ile ilgilenen öğrencilere, araştırmacılara ve yazılım geliştiricilere** yol göstermektir. Örnekler, yorumlarla açıklanmış ve her konu için **adım adım rehberlik sağlanmıştır**.

> **Hazırlayan:** [Niyazi Mert Işıksal](https://www.linkedin.com/in/niyazi-mert-isiksal-8b7920281/)

---

## 📚 **İçindekiler**
Aşağıdaki konular, OpenCV'nin temel ve ileri düzey özellikleriyle ilgilidir. Her bir konu, ayrıntılı açıklama ve örneklerle birlikte verilmiştir.

1. **Giriş ve Temel Bilgiler**
    - OpenCV Kütüphanesi Nedir?
    - Görüntü Yükleme ve Görüntü İşleme
    - Piksel Manipülasyonu
2. **Görüntü İşleme Teknikleri**
    - **Görüntü İnvert Etme** (Inverted Image)
    - **Piksel İşleme (Min, Max, Ortalama, Standart Sapma)**
    - **Piksel Eşikleme (Thresholding)**
    - **ROI (Region of Interest) Kullanımı**
3. **Filtreleme Teknikleri**
    - **Gaussian Blur (Gauss Bulanıklaştırma)**
    - **Median Blur (Medyan Bulanıklaştırma)**
    - **Bilateral Filtreleme (Bilateral Filter)**
    - **Kenarlık Algılama (Canny Edge Detection)**
4. **Görüntü Geliştirme Teknikleri**
    - **Görüntü Keskinleştirme (Sharpening)**
    - **Histogram Eşitleme (Histogram Equalization)**
    - **Görüntü Normalizasyonu**
5. **Kontur Tespiti ve Şekil Tanıma**
    - **Kontur Tespiti (cv2.findContours)**
    - **Min Alanlı Dikdörtgen (cv2.minAreaRect)**
    - **Kontur Çizgisi (cv2.drawContours)**
6. **Köşe Algılama Teknikleri**
    - **Harris Köşe Algoritması**
    - **Shi-Tomasi Köşe Tespiti**
    - **Subpixel Köşe Tespiti (cv2.cornerSubPix)**
7. **Hough Dönüşümü**
    - **Hough Çizgileri (cv2.HoughLines)**
    - **Hough Çemberleri (cv2.HoughCircles)**
8. **Optik Akış ve Hareket Tespiti**
    - **Optik Akış (cv2.calcOpticalFlowFarneback)**
    - **Arka Plan Çıkarma (Background Subtraction)**
9. **Şablon Eşleştirme (Template Matching)**
10. **Özellik Tanıma ve Anahtar Nokta Algılama**
    - **ORB (Oriented FAST and Rotated BRIEF)**
    - **SIFT (Scale Invariant Feature Transform)**
    - **BRISK (Binary Robust Invariant Scalable Keypoints)**
    - **Descriptor Matcher ile Anahtar Nokta Eşleştirme**
11. **Makine Öğrenmesi ile Nesne Tespiti**
    - **HOG ile Yaya Tespiti**
    - **SVM Kullanarak Veri Sınıflandırma**
    - **Caffe ile Obje Tespiti**
12. **Diğer Teknikler**
    - **Görüntü Maskeleme**
    - **GrabCut ile Arka Plan Çıkarma**
    - **Görüntü Boyutlandırma (Resize)**
    - **Görüntü Döndürme ve Dönüşüm (Affine Transformation)**
    - **Video İşleme**
    - **Optik Akış ile Hareket Tespiti**
13. **Veri Görselleştirme ve Histogramlar**
    - **Histogram Grafikleri**
    - **Histogram Karşılaştırma**
14. **Derin Öğrenme ile Nesne Tespiti**
    - **DNN ile Nesne Tespiti**
    - **YOLO ve SSD ile Nesne Tespiti**
    - **Caffe ile Model Yükleme ve Kullanma**

---

## 🚀 **Nasıl Kullanılır?**
Bu kılavuzu **kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin**.

### 📦 **Gereksinimler**
Bu kılavuzun çalıştırılması için aşağıdaki kütüphaneler gereklidir:
- **OpenCV** (`cv2`)
- **NumPy** (`numpy`)
- **Matplotlib** (Opsiyonel)
- **scikit-image** (Opsiyonel)

### ⚙️ **Kurulum**
1. Python 3'ü bilgisayarınıza kurun.
2. Aşağıdaki komutu terminalde çalıştırarak gerekli kütüphaneleri kurun:
    ```bash
    pip install opencv-python-headless numpy matplotlib scikit-image
    ```
