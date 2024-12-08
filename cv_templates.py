########################################################################################################################################
#OPENCV GUIDE TÜRKÇE KAYNAK
#Bu kaynak, OpenCV ve Python ile Görüntü İşleme konusunda temel bilgileri içerir.

#COMMAND + F ILE HIZLI ARAMA
#Creator: Niyazi Mert Işıksal
#Linkedin: https://www.linkedin.com/in/niyazi-mert-isiksal-8b7920281/
########################################################################################################################################


########################################################################################################################################
#KONULAR
#1. Giriş ve Temel Bilgiler

#2. Görüntü İşleme Teknikleri

#3. Filtreleme Teknikleri

#4. Görüntü Geliştirme Teknikleri

#5. Kontur Tespiti ve Şekil Tanıma

#6. Köşe Algılama Teknikleri

#7. Hough Dönüşümü

#8. Şablon Eşleştirme (Template Matching)

#9. Özellik Tanıma ve Anahtar Nokta Algılama

#10. Diğer Teknikler ve Uygulamalar

#11. Veri Görselleştirme ve Histogramlar

#12. Arka Plan Çıkarma ve Hareket Tespiti

#13. Makine Öğrenmesi ile Nesne Tespiti

#14. Morfolojik İşlemler
########################################################################################################################################



########################################################################################################################################
#NOTLAR
########################################################################################################################################

#OpenCV, görüntü işleme ve makine öğrenmesi uygulamaları için yaygın olarak kullanılan bir kütüphanedir.
#OpenCV, C++, Python, Java ve MATLAB gibi dillerde kullanılabilir.
#Görüntü işleme, bir görüntüdeki piksel değerlerini değiştirme ve görüntü üzerinde farklı işlemler yapma sürecidir.

#Görüntü İşleme Teknikleri
#Aşağıda, basit bir örnek verilmiştir.
""""
#INVERTED IMAGE ILE ORNEK
img = cv2.imread(path)
cv2.namedWindow("original", cv2.WINDOW_NORMAL)
cv2.imshow("original", img)
cv2.waitKey(0)

m1 = np.copy(img)
h, w, ch = img.shape
print(f"Height: {h}, Width: {w}, Channels: {ch}")

for row in range(h):
    for col in range(w):
        b, g, r = img[row, col]
        b = 255 - b
        g = 255 - g
        r = 255 - r
        m1[row, col] = [b, g, r]

cv2.imshow("inverted", m1)
cv2.waitKey(0)
"""

#Görüntünün en küçük (minimum) ve en büyük (maksimum) piksel değerlerini bulunması.
    #Görüntünün ortalamasını (mean) ve standart sapmasını (stddev) hesaplar.
"""
#Piksel değerleri 100'den küçük olan pikselleri sıfırlamak (siyah yapmak).
import cv2
import numpy as np

path = "Ekran Resmi 2024-12-05 18.47.48.png"
src=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
min_value,max_value,min_loc,max_loc=cv2.minMaxLoc(src)
print("min_value:",min_value)
print("max_value:",max_value)

means,stddev=cv2.meanStdDev(src)
print("means:",means)
print("stddev:",stddev)

src[np.where(src<100)]=0
cv2.imshow("src",src)
cv2.waitKey(0)
"""

#Görüntü Filtreleme İşlemleri
    #Görüntü filtreleme, bir görüntüdeki gürültüyü azaltmak ve görüntüyü daha net hale getirmek için kullanılan bir işlemdir.
    #Görüntü filtreleme işlemleri, görüntü işleme uygulamalarında yaygın olarak kullanılır.
"""
#FILTRELEME ORNEK
import cv2
import numpy as np

capture=cv2.VideoCapture(0)

height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
width=capture.get(cv2.CAP_PROP_FRAME_WIDTH)
count=capture.get(cv2.CAP_PROP_FRAME_COUNT)
fps=capture.get(cv2.CAP_PROP_FPS)

def process(image,opt=1):
    dst=None
    if opt==0:
        dst=cv2.bitwise_not(image)
    elif opt==1:
        dst=cv2.GaussianBlur(image, (0,0),15)
    elif opt==2:
        dst=cv2.Canny(image, 100, 200)
    elif opt==3:
        dst=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    elif opt==4:
        dst=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    elif opt==5:
        dst=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    return dst
    
index=5

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    #ret: Kameradan görüntü alınıp alınmadığını (başarılı/başarısız) döner.
    #frame: Kameradan alınan görüntü.
    if ret is True:
        cv2.imshow("video-input", frame)
        c = cv2.waitKey(50)
        if c >= 49:
            index = c - 49
            #Tuşun ASCII kodu 49 veya daha büyükse, index güncellenir.
            #Bu, process fonksiyonundaki opt seçimini değiştirir.
        result = process(frame, index)
        cv2.imshow("result", result)
        #İşlenmiş görüntüyü farklı bir pencerede (result) gösterir.
        if c == 27:
            #Eğer kullanıcı Esc (ASCII kodu 27) tuşuna basarsa, döngü kırılır ve işlem durdurulur.
            break
    else:
        break

cv2.waitKey(1)
# Kamerayı ve pencereleri serbest bırak
capture.release()
cv2.destroyAllWindows()
"""

#Görüntü işleme bağlamında normalleştirme
    #bir görüntünün piksel değerlerini belirli bir aralıkta ölçeklemek (örneğin, 0-255)
    #ve bu aralığa göre yeniden dağıtmak anlamına gelir.
    #Bu işlem, görüntünün kontrastını artırır ve görüntü üzerinde daha iyi analiz yapılmasını sağlar.
"""
#Rectangle Fonsksiyonunun Kullanımı
import cv2 as cv
cv.rectangle(src, pt, (pt[0] + tw, pt[1] + th), (255, 0, 0), 1, 8, 0)
        #src: Giriş görüntüsü.
        #pt: Dikdörtgenin sol üst köşesinin koordinatları.
        #(pt[0] + tw, pt[1] + th): Dikdörtgenin sağ alt köşesinin koordinatları.
        #(255, 0, 0): Dikdörtgenin rengi (BGR formatında).
        #1: Dikdörtgenin kalınlığı.
        #8: Dikdörtgenin kenar tipi.
        #0: Dikdörtgenin kenar tipi.
        #(pt[0] + tw, pt[1] + th):
            #Dikdörtgenin sağ alt köşesinin koordinatları (x, y).
            #Bu, sol üst köşeden itibaren genişlik (tw) ve yükseklik (th) kadar piksel ilerleyerek hesaplanır.
            #Örneğin:
            #Eğer tw = 100 ve th = 50 ise, sağ alt köşe (50 + 100, 50 + 50) → (150, 100) olacaktır.
"""

#Resize:
    #orijinal görüntü çok büyük, orijinal görüntünün çözünürlüğünü azaltır." ifadesi, görüntü yeniden boyutlandırma (resize) işlemini açıklar.
    #cv2.resize(src, dsize=(width, height), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    #src: Giriş görüntüsü.
    #dsize: Yeni boyut (genişlik, yükseklik).
    #fx: Genişlik ölçek faktörü.
    #fy: Yükseklik ölçek faktörü.
    #interpolation: Yeniden boyutlandırma algoritması.
"""
#RESİZE İŞLEMİ
import cv2
import numpy as np

image = cv2.imread("city-seen-from-afar.jpg")
resized = cv2.resize(image, (800, 600), interpolation=cv2.INTER_LINEAR)

cv2.imshow("Resized Image", resized)
cv2.waitKey(0)
"""

#GÜRÜLTÜ EKLEME
    #Görüntü işleme uygulamalarında, gürültü ekleme işlemi, bir görüntüye rastgele piksel değerleri eklemek anlamına gelir.
"""
#GÜRÜLTÜ EKLEME
import cv2 as cv
import numpy as np

image=cv.imread("city-seen-from-afar.jpg")
def add_pepper_salt(image):
    h, w = image.shape[:2]
    amount = 0.02
    nums=10000
    rows=np.random.randint(0,h,nums)
    cols=np.random.randint(0,w,nums)
    for i in range(nums):
        if i%2 ==1:
            image[rows[i],cols[i]]=(255,255,255)
        else:
            image[rows[i],cols[i]]=(0,0,0)
    return image

h,w = image.shape[:2]
copy=np.copy(image)
copy=add_pepper_salt(copy)
copy=add_pepper_salt(copy)

result=np.zeros((h,2*w,3),dtype=np.uint8)
result[0:h,0:w,:]=image
result[0:h,w:2*w,:]=copy

cv.imshow("result",result)
cv.waitKey(0)
"""

#GÖRÜNTÜ KESKİNLEŞTİRME
    #Görüntü keskinleştirme, bir görüntüdeki kenarları ve detayları vurgulamak için kullanılan bir işlemdir.
    #Bu işlem, görüntüdeki bulanıklığı azaltır ve görüntüyü daha net hale getirir.
    #Görüntü keskinleştirme, genellikle bir kenar algılama algoritması ile birlikte kullanılır
    #ve görüntüdeki kenarları daha belirgin hale getirmek için kullanılır.
"""
#GÖRÜNTÜ KESKİNLEŞTİRME
import cv2 as cv
import numpy as np

image=cv.imread("city-seen-from-afar.jpg")

sharpen_op=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
sharpen_image=cv.filter2D(image,-1,sharpen_op)
sharpen_image=cv.convertScaleAbs(sharpen_image)

cv.imshow("sharpen_image",sharpen_image)
cv.waitKey(0)
"""

#SOBEL
    #Sobel Operatörü, görüntü işleme alanında kenar tespiti yapmak için kullanılan bir diferansiyel filtredir.
    #Sobel operatörü, bir görüntünün gradyanını (eğimini) hesaplayarak, görüntüdeki kenarları belirgin hale getirir.
    #Kenar tespiti, bir görüntüdeki parlaklık değişimlerini tespit etmeye dayanır.
    #Sobel, bu amaçla türevleri hesaplar ve bu türevlerden kenar bölgelerini çıkarır.

#cv2.Sobel():
    #src: Girdi görüntüsü (genelde gri tonlamalı olur).
    #ddepth: Çıktı görüntüsünün derinliği. Genellikle cv2.CV_64F kullanılır.
    #dx: x yönündeki türevin derecesi (Sobel X için dx=1).
    #dy: y yönündeki türevin derecesi (Sobel Y için dy=1).
    #ksize: Çekirdek boyutu. (Genellikle 3 veya 5 kullanılır).

#magnitude ve phase metodları, görüntü işleme ve sinyal işleme alanlarında sıklıkla kullanılan gradyan bilgilerini analiz etmek için kullanılır.
#Bunlar, bir görüntüdeki gradyan yönü ve büyüklüğünü (magnitude and phase) hesaplamak için kullanılır.

#cv2.magnitude():
    #x: x yönündeki türev.
    #y: y yönündeki türev.
    #magnitude: x ve y türevlerinin büyüklüğünü hesaplar.

#cv2.phase():
    #x: x yönündeki türev.
    #y: y yönündeki türev.
    #angle: x ve y türevlerinin açısını hesaplar.
"""
#SOBEL VE GRADYANLAR İÇİN ÖRNEK
import cv2
import numpy as np
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# Sobel gradyanları
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # X yönünde türev
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Y yönünde türev

# Büyüklüğü (magnitude) ve yönü (phase) hesapla
magnitude, phase = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=True)

# Görüntüleri göster
cv2.imshow("Magnitude", cv2.convertScaleAbs(magnitude))
cv2.imshow("Phase", phase)  # Faz, genelde görselleştirme için normalize edilir
cv2.waitKey(0)
"""

#TRESHOLD ILE ESIK DEGERİ
    #Eşikleme, bir görüntüdeki piksel değerlerini belirli bir eşik değerine göre sınıflandırmak için kullanılan bir işlemdir.
    #cv2.threshold(src, thresh, maxval, type, dst=None)
    #src: Giriş görüntüsü.
    #thresh: Eşik değeri.
    #maxval: Maksimum değer.
    #type: Eşikleme türü.
    #dst: Çıktı görüntüsü.
"""
capture=cv2.VideoCapture(0)
height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
width=capture.get(cv2.CAP_PROP_FRAME_WIDTH)
count=capture.get(cv2.CAP_PROP_FRAME_COUNT)
fps=capture.get(cv2.CAP_PROP_FPS)

def process(image,opt=1):
    if opt==1:
        return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    elif opt==2:
        return cv2.GaussianBlur(image,(11,11),0)
    elif opt==3:
        _,esik=cv2.threshold(image,240,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return esik
    elif opt==4:
        kontur,_=cv2.findContours(image.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        el_kontur=max(kontur,key=cv2.contourArea)
        return el_kontur
    elif opt==5:
        kon_kabuk=cv2.convexHull(image,returnPoints=False)
        defekt=cv2.convexityDefects(image,kon_kabuk)
        return defekt
    else:
        return image
"""

#DescriptorMatcher_create() bir tanımlayıcı (descriptor) oluşturmaz,
    # mevcut tanımlayıcıları eşleştirmek için bir eşleştirici (matcher) oluşturur.
    # Bu eşleştirici, iki farklı görüntüdeki anahtar noktaları (keypoints) karşılaştırmak ve eşlemek için kullanılır.

#Brute-Force (BF) ve SIFT ile Görüntü Eşleştirme
    #SIFT (Scale-Invariant Feature Transform), BRUTEFORCE Matcher ve anahtar noktalar (keypoints),
    #İki görüntü arasındaki benzerlikleri tespit etmek ve görüntüleri karşılaştırmak için kullanılır.
"""
#DESCRİPTOR MATCHER VE BF ILE ORNEK
import cv2
image1 = cv2.imread('image1.jpg', 0)
image2 = cv2.imread('image2.jpg', 0)

# SIFT dedektörünü oluştur
sift = cv2.SIFT_create()

# Her iki görüntüdeki anahtar noktaları ve tanımlayıcıları bul
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Brute-Force eşleştiriciyi oluştur
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)

# Tanımlayıcıları eşleştir
matches = matcher.match(descriptors1, descriptors2)

# Eşleşmeleri mesafeye göre sırala
matches = sorted(matches, key=lambda x: x.distance)

# İlk 20 eşleşmeyi çiz
result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:20], None, flags=2)

# Sonucu göster
cv2.imshow('Brute-Force Eşleşmeleri', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#BILATERAL FILTRE ILE GORUNTU IYILESTIRME
    #Bilateral filtre, bir görüntüdeki kenarları koruyarak gürültüyü azaltmak için kullanılan bir filtreleme tekniğidir.
    #edgePreservingFilter() fonksiyonu, bir görüntüdeki kenarları koruyarak gürültüyü azaltmak için kullanılan bir filtreleme tekniğidir.
    #cv2.edgePreservingFilter(src, sigma_s=60, sigma_r=0.4, flags=cv2.RECURS_FILTER)
    #src: Giriş görüntüsü.
    #sigma_s: Uzamsal filtreleme parametresi.
    #sigma_r: Yoğunluk filtreleme parametresi.
    #flags: Filtreleme bayrakları.
"""
#BİLATERAL FİLTRE GÖRÜNTÜ İYİLEŞTİRME / EDGE PRESEVİNG FİLTER
import cv2
import numpy as np

src=cv2.imread("city-seen-from-afar.jpg")
cv2.imshow("hgf",src)
cv2.waitKey(0)

h, w = src.shape[:2]

dst=cv2.bilateralFilter(src,15,800,80)

result=np.zeros((h,2*w,3),dtype=np.uint8)
result[0:h,0:w,:]=src
result[0:h,w:2*w,:]=dst

cv2.imshow("result",result)
cv2.waitKey(0)

epf=cv2.edge_preserving_filter(src, sigma_s=60, sigma_r=0.4, flags=cv2.RECURS_FILTER)
"""

#np.histogram(), bir veri kümesinin histogramını hesaplamak için kullanılan bir fonksiyondur.
    #hist, bins = np.histogram(data, bins=10, range=(0, 1))
    #a: Giriş veri kümesi.
    #bins: Histogram sütunları arasındaki sınırlar.
    #range: Histogram aralığı.
    #hist: Hesaplanan histogram.
    #np.histogram().astype("float"), histogramı float veri türüne dönüştürür.
"""
#HISTOGRAM ORNEK
import cv2
import numpy as np
from matplotlib import pyplot as plt

src = cv2.imread("city-seen-from-afar.jpg")

# Görüntünün başarıyla yüklendiğini kontrol et
if src is None:
    print("Hata: Görsel yüklenemedi! Dosya yolunu kontrol edin.")
    exit()

def custom_hist(gray):
    h, w = gray.shape
    hist = np.zeros([256], dtype=np.int32)
    for row in range(h):
        for col in range(w):
            pv = gray[row, col]
            hist[pv] += 1
            #hist: 256 uzunluğunda bir dizi (her piksel yoğunluğu için bir alan) oluşturulur.
            #İki döngüyle görüntüdeki her piksel dolaşılır. Piksel değeri (pv) histogram dizisinde karşılık gelen hücreyi artırır.

    y_pos = np.arange(0, 256, 1, dtype=np.int32)
    plt.bar(y_pos, hist, align="center", color="r", alpha=0.5)
    plt.xticks(y_pos, y_pos)
    plt.ylabel("Frequency")
    plt.title("Histogram")
    plt.show()

def image_hist(image):
    cv2.imshow("input", image)

    if len(image.shape) == 2:
        # Gri tonlamalı görüntü için histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist, color="gray")
        plt.xlim([0, 256])
    else:
        # Renkli görüntü için her renk kanalı
        color = ("blue", "green", "red")
        for i, color in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
            plt.xlim([0, 256])

    plt.show()

# Gri tonlamaya çevirme
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# Elle histogram oluşturma
custom_hist(gray)
image_hist(gray)

# Renkli histogram
image_hist(src)

# Histogram eşitleme ve eşitlenmiş histogram
cst = cv2.equalizeHist(gray)
if cst is not None:
    cv2.imshow("Custom Histogram Equalization", cst)
    image_hist(cst)
    custom_hist(cst)

#HİSTOGRAM KARŞILAŞTIRMA
src1=cv2.imread("Unicuslogo.png")
src2=cv2.imread("city-seen-from-afar.jpg")

hsv1=cv2.cvtColor(src1,cv2.COLOR_BGR2HSV)
hsv2=cv2.cvtColor(src2,cv2.COLOR_BGR2HSV)

hist1=cv2.calcHist([hsv1],[0,1],None,[180,256],[0,180,0,256])
cv2.normalize(hist1,hist1,0,255,cv2.NORM_MINMAX)
hist2=cv2.calcHist([hsv2],[0,1],None,[180,256],[0,180,0,256])
cv2.normalize(hist2,hist2,0,255,cv2.NORM_MINMAX)

print(cv2.compareHist(hist1,hist2,cv2.HISTCMP_CORREL))
"""

#local_binary_pattern(), bir görüntüdeki yerel ikili deseni (LBP) hesaplamak için kullanılan bir fonksiyondur.
    #image: Giriş görüntüsü.
    #P: Pikselin çevresindeki piksel sayısı.
    #R: Pikselin çevresindeki yarıçap.
    #method: LBP yöntemi (örn: "uniform").
    #LBP: Hesaplanan LBP görüntüsü.
"""
#LBP İLE YÜZ TANIMA
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# Görüntüyü yükle ve gri tonlamaya çevir
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# LBP parametreleri
radius = 1  # Komşuluk yarıçapı
n_points = 8 * radius  # Komşu piksel sayısı

# LBP hesaplama
lbp = local_binary_pattern(image, n_points, radius, method="uniform")

# Histogram oluşturma
(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

# Histogramı normalize et
hist = hist.astype("float")
hist /= hist.sum()

# Görüntü ve histogramı göster
plt.subplot(1, 2, 1)
plt.imshow(lbp, cmap="gray")
plt.title("LBP Görüntüsü")

plt.subplot(1, 2, 2)
plt.bar(np.arange(0, len(hist)), hist)
plt.title("LBP Histogramı")
plt.show()
"""

#KONTUR ÇİZGİLERİNİ BULMA
    #Kontur çizgileri, bir nesnenin sınırlarını veya şeklini belirten bir dizi noktadan oluşan bir eğridir.
"""
#KONTUR ÇİZGİLERİNİ BULMA
import cv2
import numpy as np

def treshold_demo(image):
    dst=cv2.GaussianBlur(image,(3,3),0)
    gray=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    ret,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    cv2.imshow("binary",binary)
    return binary

def canny_demo(image):
    t=100
    canny_output=cv2.Canny(image,t,t*2)
    cv2.imshow("canny_output",canny_output)
    return canny_output

src=cv2.imread("city-seen-from-afar.jpg")
cv2.imshow("input",src)
cv2.waitKey(0)

binary=treshold_demo(src)
canny=canny_demo(src)

contours, hierarchy=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#Binary görüntüde konturları (şekil sınırlarını) bulur.
for c in range(len(contours)):
    cv2.drawContours(src,contours,c,(0,0,255),2,8)
    #Konturları orijinal görüntü üzerinde çizer.

cv2.imshow("contours",src)
cv2.waitKey(0)
"""

#Harris Köşe Algoritması (cv.cornerHarris):
    #src: Giriş görüntüsü.
    #blockSize, köşe tespitinde değerlendirilecek komşuluk boyutunu belirler.
    #ksize (Sobel türev çekirdeği boyutu): Sob
    #k: Köşe tespitinde kullanılan serbest parametre.
"""
#HARRİS KÖŞE TESPİTİ
import cv2 as cv
import numpy as np

image=cv.imread("IMG_8132.png")

def harris(image):
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    dst=cv.cornerHarris(gray,2,3,0.04)
    dst_norm=np.empty(dst.shape,dtype=np.float32)
    cv.normalize(dst,dst_norm,0,255,cv.NORM_MINMAX)
    #cv.normalize: dst matrisinin değerlerini 0 ile 255 arasında ölçekle

    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j])>120:
                cv.circle(image,(j,i),5,(0,0,255),2)
                #Eğer dst_norm[i, j] > 120 (piksel yoğunluğu 120'den büyükse), bu pikselin bir köşe olduğu kabul edilir.
                #Bu değer (120), köşe tespit eşiği olarak kabul edilir. Daha yüksek bir eşik, daha az köşe tespit eder.
    return image

result=harris(image)
cv.imshow("result",result)
cv.waitKey(0)
"""

#SHI-TOMASI KÖŞE TESPİTİ
    #cv.goodFeaturesToTrack(), bir görüntüdeki köşeleri tespit etmek için kullanılan bir fonksiyondur.
    #corners= cv.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance)
    #image: Giriş görüntüsü.
    #maxCorners: Tespit edilecek köşe sayısı.
    #qualityLevel: Köşelerin kalitesi.
    #minDistance: Tespit edilen köşeler arasındaki minimum mesafe.
    #corners: Tespit edilen köşelerin koordinatları.
"""
#SHİ-TOMASİ KÖŞE TESPİTİ
import cv2 as cv
import numpy as np

image=cv.imread("IMG_8132.png")
def shi_tomasi(image):
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    corners=cv.goodFeaturesToTrack(gray,100,0.01,10)
    corners=np.int0(corners)
    for i in corners:
        x,y=i.ravel()
        cv.circle(image,(x,y),5,(0,0,255),2)
    return image

result=shi_tomasi(image)
cv.imshow("result",result)
cv.waitKey(0)
"""

#cv2.convertScaleAbs(), OpenCV'nin bir fonksiyonudur ve temel amacı bir görüntüyü ölçeklendirmek (scale),
# mutlak değerini almak (absolute value) işlemlerini yaparak, 8-bitlik bir görüntüye (0-255 aralığında) dönüştürmektir.
    #alpha: Katsayı (piksel değerlerini çarpan bir ölçek faktörü).
    #beta: Piksel değerlerine eklenen ofset (bir sabit değer ekler).
"""
#ORNEK KOD
import cv2
import numpy as np

image = cv2.imread("city-seen-from-afar.jpg")
sharpen_op = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpen_image = cv2.filter2D(image, -1, sharpen_op)
sharpen_image = cv2.convertScaleAbs(sharpen_image)

cv2.imshow("sharpen_image", sharpen_image)
cv2.waitKey(0)
"""

#cv2.cornerSubPix(), bir görüntüdeki köşelerin konumlarını daha hassas bir şekilde belirlemek için kullanılan bir fonksiyondur.
    #corners = cv2.cornerSubPix(image, corners, winSize, zeroZone, criteria)
    #corners: Köşelerin konumları.
    #winSize: Köşelerin konumlarını belirlemek için kullanılan pencere boyutu.
    #zeroZone: Sıfır bölgesi (devre dışı bırakılan bölge).
    #criteria: Döngü kriterleri.

#cv2.findChessboardCorners(), bir satranç tahtasındaki köşeleri tespit etmek için kullanılan bir fonksiyondur.
    #ret, corners = cv2.findChessboardCorners(image, patternSize, corners, flags)
    #ret: Başarılı olup olmadığını belirten bir bayrak.
    #patternSize: Satranç tahtasının boyutu (sütun, satır).
    #corners: Köşelerin konumları.
    #flags: İşlem bayrakları.
"""
#SUBPİXEL KÖŞE TESPİTİ
import cv2 as cv
import numpy as np

image=cv.imread("satranc.png")
def process(image):
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    corners=cv.goodFeaturesToTrack(gray,100,0.01,10)
    print(len(corners))
    for pt in corners:
        b= np.random.random_integers(0,255)
        g= np.random.random_integers(0,255)
        r= np.random.random_integers(0,255)
        #Amaç: Her köşe için rastgele bir BGR rengi atanır.
        x=np.int32(pt[0][0])
        y=np.int32(pt[0][1])
        #pt[0][0] ve pt[0][1]: Bu, köşenin (x, y) koordinatını verir.
        cv.circle(image,(x,y),5,(int(b),int(g),int(r)),2)

    winSize=(5,5)
    #winSize: Her köşe için çevresinde kontrol edilecek pencerenin boyutu (5x5 pencere).
    zeroZone=(-1,-1)
    #zeroZone: Sıfır bölgesi. (-1, -1) olarak ayarlanması, bölgenin devre dışı bırakılmasını sağlar.
    criteria=(cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER,40,0.001)
    #criteria: Döngü kriterleri. Bu, köşelerin hassas bir şekilde bulunmasını sağlar.
    #criteria: Tekrarlama kriterleri.
    #cv.TERM_CRITERIA_EPS: Algoritma, hata belirli bir seviyeye düştüğünde durur.
    corners=cv.cornerSubPix(gray,corners,winSize,zeroZone,criteria)
    #Algılanan köşelerin doğruluğunu artırır

    for i in range(corners.shape[0]):
        print(corners[i,0])
        #Amaç: Subpixel seviyesindeki her köşenin (x, y) koordinatlarını ekrana yazdırır.
    return image

result=process(image)
cv.imshow("result",result)
cv.waitKey(0)
"""

#cv2.HOGDescriptor(), Histogram of Oriented Gradients (HOG) tanımlayıcısını oluşturmak için kullanılan bir sınıftır.
    #setSVMDetector(): HOG tanımlayıcısını oluşturmak için kullanılan bir yöntemdir.
    #detectMultiScale(): HOG tanımlayıcısını kullanarak nesneleri algılamak için kullanılan bir yöntemdir.

#hog.detectMultiScale():
    #Histogram of Oriented Gradients (HOG) kullanarak,
    #Bir görüntüdeki farklı boyutlardaki nesneleri algılamak için kullanılan bir fonksiyondur.
    #Bu fonksiyon, nesnelerin konumunu ve boyutunu döndürür.
    #winStride: Tespit sırasında kullanılan kaydırma adımı (örn: (4, 4)).
    #padding: Tespit sırasında kullanılan dolgu boyutu (örn: (8, 8)).
    #scale: Tespit sırasında kullanılan ölçek faktörü (örn: 1.05).
    #useMeanshiftGrouping: Tespit edilen bölgeleri gruplamak için kullanılan yöntem.

"""
#HOG İLE YAYA TESPTİ

import cv2 as cv
import numpy as np

image=cv.imread("satranc.png")
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
rects,weights=hog.detectMultiScale(image,winStride=(4,4),padding=(8,8),scale=1.05)

for (x,y,w,h) in rects:
    cv.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)

cv.imshow("image",image)
cv.waitKey(0)
"""

#HOUGH ÇİZGİLERİ
    #cv2.HoughLines(), bir görüntüdeki doğru çizgileri tespit etmek için kullanılan bir fonksiyondur.
    #lines = cv2.HoughLines(image, rho, theta, threshold)
    #image: Giriş görüntüsü.
    #rho: r parametresi.
    #theta: θ parametresi.
    #threshold: Eşik değeri.
    #lines: Tespit edilen çizgilerin koordinatları.
"""
#HOUGH ÇİZGİLERİNİ BULMA
import cv2
import numpy as np

def hough_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        for rho, theta in lines[:, 0]:
            #Eğer lines boş değilse (yani bir çizgi bulunmuşsa), her bir çizgi üzerinde döngü yapar.
            a = np.cos(theta)
            #a = cos(theta): X eksenindeki yön vektörü.
            b = np.sin(theta)
            x0 = a * rho #Çizginin orijine olan mesafesi (rho) ve açısı (theta) kullanılarak hesaplanan bir nokta.
            y0 = b * rho
            #Bu hesaplama, çizgiyi görüntüde uzun bir çizgi olarak çizebilmek için yapılır.
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            #(x1, y1) ve (x2, y2): Çizginin iki uç noktasıdır.
            #1000: Çizginin görüntü sınırları boyunca uzanmasını sağlar (uzun bir çizgi çizer).
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image

scr=cv2.imread("city-seen-from-afar.jpg")
son=hough_lines(scr)
cv2.imshow("son",son)
cv2.waitKey(0)
"""

#cv.SIFT_create() (Scale-Invariant Feature Transform)
    #Görüntüde ölçek ve döndürmeye karşı dayanıklı kilit noktaları tespit eder.
    #Her kilit noktaya ait özellik vektörleri (descriptors) oluşturur.
    #Bu vektörler, kilit noktaları tanımlamak için kullanılır.

#cv2.warpAffine()
    #Görüntüde geometrik dönüşümler yapmak için kullanılan bir fonksiyondur.
    #cv2.warpAffine(image, M, (width, height))
    #image: Giriş görüntüsü.
    #M: Dönüşüm matrisi.
    #(width, height): Çıktı görüntüsünün boyutu.
"""
#SHIFT-CREATE ORNEK
import cv2
import numpy as np

# Görüntüyü yükle
image = cv2.imread('example.jpg')

# Kaydırma matrisini oluştur (50 piksel sağa, 30 piksel aşağıya)
tx = 50  # x ekseni boyunca kaydırma
ty = 30  # y ekseni boyunca kaydırma
M = np.float32([[1, 0, tx], [0, 1, ty]])

# Görüntüyü kaydır
shifted_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

# Sonucu göster
cv2.imshow('Orijinal Görüntü', image)
cv2.imshow('Kaydırılmış Görüntü', shifted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

img=cv2.imread("Ekran Resmi 2024-12-05 18.47.48.png")
rows=img.shape[0]
cols=img.shape[1]

#M=np.float32([[1,0,300],[0,1,90]])
M=cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst=cv2.warpAffine(img,M,(cols,rows))
res=cv2.resize(img,None,fx=0.2,fy=0.2,interpolation=cv2.INTER_CUBIC)

cv2.imshow("Resized Image",res)
cv2.imshow("Original Image",img)
cv2.imshow("Shifted Image",dst)
cv2.waitKey(0)#herhangi bir tuşa basana kadar
"""

#ORB Algoritması
    #Oriented FAST and Rotated BRIEF (ORB), görüntü eşleştirme ve nesne tanıma için kullanılan bir algoritmadır.
    #ORB, FAST (Features from Accelerated Segment Test) ve BRIEF (Binary Robust Independent Elementary Features) algoritmalarını birleştirir.
    #Bu algoritma, görüntüdeki anahtar noktaları (keypoints) algılar ve bu noktalar için tanımlayıcıları (descriptors) hesaplar.
    #Bu tanımlayıcılar, görüntüdeki nesneleri tanımlamak için kullanılır.
    #ORB, SHIFT'e göre daha hızlı bir işlemdir.

#detectAndCompute():
    #Anahtar noktaları algılar ve bu noktalar için tanımlayıcıları hesaplar.
    #Bu fonksiyon, hem anahtar noktaların yerini tespit eder hem de bu noktalar için özellik vektörlerini oluşturur.
    #keypoints, descriptors = sift.detectAndCompute(image, None)
    #keypoints: Algılanan anahtar noktalar.
    #descriptors: Anahtar noktaları tanımlamak için kullanılan özellik vektörleri.
    #mask=None (maske): Anahtar noktaların tespit edileceği alanı belirlemek için kullanılır.

#cv2.drawMatches():
    #İki görüntü arasındaki eşleşmeleri çizer.
    #Bu fonksiyon, iki görüntü arasındaki eşleşmeleri görselleştirmek için kullanılır.
"""
#ORB-DETECTANDCOMPUTE-DRAWMATCHES ORNEK
import cv2

image1 = cv2.imread('image1.jpg', 0)  # Gri tonlamalı
image2 = cv2.imread('image2.jpg', 0)  # Gri tonlamalı

orb = cv2.ORB_create()

# Her iki görüntüdeki anahtar noktaları ve tanımlayıcıları bul
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# BFMatcher ile karşılaştırma yap
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Tanımlayıcıları karşılaştır
matches = bf.match(descriptors1, descriptors2)

# Eşleşmeleri mesafeye göre sırala
matches = sorted(matches, key=lambda x: x.distance)

# İlk 20 eşleşmeyi çiz
result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:20], None, flags=2)

cv2.imshow('ORB Eşleşmeleri', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#Template Matching
    #cv2.matchTemplate(src, tpl, method, result=None, mask=None)
    #src: Giriş görüntüsü.
    #tpl: Şablon görüntüsü.
    #method: Eşleştirme yöntemi (örn: cv2.TM_CCOEFF_NORMED).
    #result: Eşleştirme sonucu.
    #mask: Eşleştirme alanını sınırlamak için kullanılan maske.
"""
#TEMPLATE MATCİNG İLE NESNE TESPİTİ
import cv2 as cv
import numpy as np
def template_demo():
    # Read the template and source images
    tpl = cv.imread("Ekran Resmi 2024-12-07 17.57.29.png")
    src = cv.imread("city-seen-from-afar.jpg")

    # Check if images are loaded properly
    if tpl is None:
        print("Error: Template image not found.")
        return
    if src is None:
        print("Error: Source image not found.")
        return

    # Display the images
    cv.imshow("Template", tpl)
    cv.imshow("Source", src)

    # Get dimensions of the template
    h, w = tpl.shape[:2]

    # Perform template matching
    result = cv.matchTemplate(src, tpl, cv.TM_CCOEFF_NORMED)
    #cv.matchTemplate: Şablonun hedef görüntüde nerede olduğunu bulur.
    cv.imshow("Matching Result", result)

    t = 0.98
    loc = np.where(result >= t)
    for pt in zip(*loc[::-1]):
        cv.rectangle(src, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    cv.imshow("Detected Matches", src)
template_demo()
cv.waitKey(0)
"""

#Maskeleme
    #Bir görüntüde belirli bir bölgeyi seçmek için kullanılan bir tekniktir.
    #Bu teknik, görüntüdeki belirli bir bölgeyi işlemek ve diğer bölümleri göz ardı etmek için kullanılır.
    #G(x,y) = I(x,y) . M(x,y)
"""
#ÖRNEK MASKELEME KODU
import cv2
import numpy as np

# Görüntüyü yükle
image = cv2.imread('example.jpg')

# Görüntü boyutunu al
height, width = image.shape[:2]

# Boş bir maske (tamamen siyah)
mask = np.zeros((height, width), dtype=np.uint8)

# Dairenin merkezi ve yarıçapı
center = (width // 2, height // 2)  # Görüntü ortası
radius = 100  # Yarıçap

# Daireyi maske üzerine çiz
cv2.circle(mask, center, radius, 255, -1)  # -1, daireyi tamamen doldurur

# Maske ile görüntüyü çarp
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Görüntüleri göster
cv2.imshow('Orijinal Görüntü', image)
cv2.imshow('Maske', mask)
cv2.imshow('Maskeleme Sonucu', masked_image)
cv2.waitKey(0)
"""

#cv2.createBackgroundSubtractorMOG2():
    #Görüntüdeki arka planı çıkarmak için bir MOG2 (Gauss Karışım Modeli) nesnesi oluşturur.
    #history: Arka plan modelini oluşturmak için kullanılan kare sayısı.
    #varThreshold: Piksel değişimini "ön plan mı, arka plan mı" olarak sınıflandırmak için eşik değeri.
    #detectShadows: Gölge alanlarını algılayıp maske üzerinde ayırt etmesini sağlar.

#cv2.getStructuringElement():
    #OpenCV kütüphanesinde kullanılan, morfolojik işlemler (erozyon, genişletme, açma, kapama vb.)
    #için kullanılan yapılandırma elemanını (structuring element) oluşturan bir fonksiyondur.
    #Bu yapılandırma elemanı, genellikle bir dikdörtgen, elips veya çapraz (cross) şekline sahip bir çekirdek (kernel) matrisidir.
    #Bu çekirdek (kernel), görüntüde morfolojik işlemleri gerçekleştirirken hangi piksellerin etkilenmesi gerektiğini belirler.
    #Yapılandırma elemanının boyutu ve şekli, görüntüdeki etkisini doğrudan etkiler.
    #shape: Yapı elemanının şekli (cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS).
    #size: Yapı elemanının boyutu (genişlik ve yükseklik).
    #anchor: Yapı elemanının merkez noktası.

#cv2.findContours():
    #Görüntüdeki konturları (şekil sınırlarını) bulmak için kullanılan bir fonksiyondur.
    #mode: Kontur algılama modu (örn: cv2.RETR_EXTERNAL, cv2.RETR_LIST).
    #method: Kontur yaklaşım yöntemi (örn: cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_NONE).
    #contours: Bulunan konturların listesi.
    #hierarchy: Kontur hiyerarşisi. Yani konturlar arasındaki ilişkiyi belirtir.

#cv2.minAreaRect():
    #cv2.minAreaRect, bir kontur çevresine yerleştirilebilecek minimum alanlı dikdörtgeni (minimum bounding rectangle) bulur.
    #Bu dikdörtgen, konturu çevreleyen ve alanı en küçük olan döndürülmüş (rotated) bir dikdörtgendir.
    #Yani, konturun etrafına "dönen" bir dikdörtgen çizer ve bu dikdörtgenin eğim açısını (rotation angle) de döndürür.
    #Bu fonksiyon, özellikle bir şekli çevreleyen dikdörtgenin konumunu, boyutunu ve açısını bulmak için kullanılır.
    #Örneğin, nesnelerin dönüş açısını veya şekil özelliklerini bulmak için idealdir.
    #contour: Kontur verisi.

#cv2.cv2.calcOpticalFlowFarneback():
    #İki ardışık görüntü arasındaki optik akışı (optical flow) hesaplamak için kullanılan OpenCV fonksiyonudur.
    #Optik akış, bir nesnenin bir görüntüden diğerine olan hareketini piksel düzeyinde izlemek için kullanılan bir yöntemdir.
    #Bu yöntem, her bir pikselin iki görüntü arasındaki nasıl hareket ettiğini bulur. Bu hareket, bir hız vektörü ile temsil edilir.
    #Fonksiyon, her piksel için bir (dx, dy) hareket vektörü döndürür.
"""
#ÜSTTEKİ 5 FONKSİYON ILE ARKA PLAN ÇIKARMA ÖRNEĞİ 1
import cv2
import numpy as np

cap=cv2.VideoCapture("godfather korna.mp4")
fbgb=cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=50,detectShadows=True)

def process(image):
    mask=fbgb.apply(image)
    line=cv2.getStructuringElement(cv2.MORPH_RECT,(1,5),(-1,-1))
    #Dikdörtgen bir yapı elemanı oluşturur.
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,line)
    #MorphologyEx fonksiyonu, morfolojik işlemleri uygular.
    cv2.imshow("mask",mask)
    contours, hierachy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #Maske üzerinde konturları (şekil sınırlarını) bulur.
    #cv2.RETR_EXTERNAL: Sadece dış konturları alır.
    #cv2.CHAIN_APPROX_SIMPLE: Konturun sadece başlangıç ve bitiş noktalarını alır.
    for c in range(len(contours)):
    #Tüm konturlar üzerinde bir döngü başlatır.
        area=cv2.contourArea(contours[c])
        #cv2.contourArea: Konturun alanını (piksel cinsinden) hesaplar.
        if area<150:
            continue
        #Eşik: Eğer kontur alanı 150 pikselden küçükse, bu kontur göz ardı edilir. Bu işlem, küçük gürültüleri filtrelemek için yapılır.
        rect=cv2.minAreaRect(contours[c])
        #cv2.minAreaRect: Konturun etrafına sığacak en küçük dikdörtgeni hesaplar.
        cv2.ellipse(image,rect,(0,255,0),2,8)
        #cv2.ellipse: Dikdörtgenin etrafına bir elips çizer.
        cv2.circle(image,(np.int32(rect[0][0]),np.int32(rect[0][1])),2,(255,0,0),2,8,0)
        #cv2.circle: Elipsin merkezine bir daire çizer.
    return image

while True:
    ret,frame=cap.read()
    if ret is False:
        break
    result=process(frame)
    cv2.imshow("result",result)
    if cv2.waitKey(30)&0xFF==27:
        break

#ARKA PLAN ÇIKARMA ÖĞRNEĞİ 2

cap=cv2.VideoCapture("godfather korna.mp4")
fbgb=cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=50,detectShadows=True)
#history=100: Arka plan modelini oluşturmak için kullanılan kare sayısını belirtir.
#varThreshold=50: Piksel değişimini "ön plan mı, arka plan mı" olarak sınıflandırmak için eşik değeri.
#detectShadows=True: Gölgeleri algılayıp maske üzerinde ayırt etmesini sağlar (gölge alanlar genellikle gri olarak işaretlenir).
while True:
#Sonsuz bir döngü başlatır. Bu döngü her kareyi işlemeye devam eder.
    ret,frame=cap.read() #ret true ya da false, frame ise kareyi döndürür.
    fgmask=fbgb.apply(frame) #fgmask: Ön plan maskesi (görüntüdeki ön plan nesnelerini belirler).
    background=fbgb.getBackgroundImage() # Modelden tahmin edilen arka plan görüntüsünü döndürür.
    cv2.imshow("frame",frame)
    cv2.imshow("fgmask",fgmask)
    cv2.imshow("background",background)
    if cv2.waitKey(30)&0xFF==27:
        break
cap.release()
"""

#ROI (Region of Interest):
    #ROI, bir görüntüde belirli bir bölgeyi seçmek için kullanılan bir tekniktir.
"""
#ROI ÖRNEK
import cv2
import numpy as np

src=cv2.imread("Ekran Resmi 2024-12-05 18.47.48.png")
print(src.shape[:2])
h, w= src.shape[:2]
img=src.copy()

roi=img[300:750,950:1300,:]
print(roi.shape[:2])

#roi_resized = cv2.resize(roi, (26, 50))  # Genişlik: 26, Yükseklik: 50
#img[0:50, 0:26, :] = roi_resized

img[0:600, 0:300, :] = roi

res=cv2.resize(roi,None,fx=0.3,fy=0.3,interpolation=cv2.INTER_CUBIC)
print(res.shape[:2])
img[0:135,0:38,:]=res

#cv2.imshow("Resized ROI",roi_resized)
cv2.imshow("Resized",res)
cv2.imshow("Image",img)
cv2.imshow("ROI",roi)
cv2.waitKey(0)

#img[0:50, 0:26, :] = roi
#Hedef bölge: (50, 26, 3) → 50 satır, 26 sütun, 3 renk kanalı.
#Kaynak bölge (ROI): (450, 126, 3) → 450 satır, 126 sütun, 3 renk kanalı.
"""

#GRABCUT
    #cv2.grabCut(), bir görüntüdeki nesneleri ve arka planı ayırmak için kullanılan bir algoritmadır.
    #Bu algoritma, bir dikdörtgen (rectangle) veya maske (mask) kullanarak nesneleri ve arka planı ayırmak için kullanılır.
    #cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode)
    #mask: İşlem yapılacak alanı belirten maske.
    #rect: İşlem yapılacak alanı belirten dikdörtgen.
    #bgdModel: Arka plan modeli.
    #fgdModel: Ön plan modeli.
    #iterCount: İterasyon sayısı.
    #mode: İşlem modu (cv2.GC_INIT_WITH_RECT, cv2.GC_INIT_WITH_MASK).
"""
#GRAPCUT VE ROI ÖRNEK
import cv2 as cv
import numpy as np

# Görseli yükleme ve yeniden boyutlandırma
src = cv.imread("city-seen-from-afar.jpg")
if src is None:
    print("Hata: Görsel yüklenemedi! Lütfen dosya yolunu kontrol edin.")
    exit()

# Görüntüyü yeniden boyutlandır
src = cv.resize(src, (0, 0), fx=0.5, fy=0.5)
#Bu, büyük görüntülerle çalışırken daha hızlı işlem yapılmasını sağlar.

# Kullanıcıdan ROI seçmesini iste
r = cv.selectROI("Select ROI", src, showCrosshair=True, fromCenter=False)
#Sonuç: r, seçilen bölgeyi (x, y, w, h) formatında döndürür

# Eğer kullanıcı bir ROI seçmezse programdan çık
if r == (0, 0, 0, 0):
    print("Hata: ROI seçilmedi!")
    cv.destroyAllWindows()
    exit()

# ROI'yi kırp
roi = src[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
#Görüntüden seçilen ROI'yi kırpar (keser).
#[y:y+h, x:x+w]:
    #int(r[1]): ROI'nin başlangıç y koordinatı.
    #int(r[1] + r[3]): ROI'nin bitiş y koordinatı.
    #int(r[0]): ROI'nin başlangıç x koordinatı.
    #int(r[0] + r[2]): ROI'nin bitiş x koordinatı.


# Görüntüyü kopyala ve ROI'yi dikdörtgenle işaretle
img = src.copy()
cv.rectangle(img, (int(r[0]), int(r[1])), (int(r[0] + r[2]), int(r[1] + r[3])), (0, 255, 0), 2)

# Maske ve GrabCut için modeller oluştur
#Görselin yüksekliği ve genişliği kadar bir maske dizisi oluşturulur.
#Bu maske GrabCut algoritması için kullanılır

mask = np.zeros(src.shape[:2], np.uint8)

#bgdmodel ve fgdmodel:
    #GrabCut algoritmasında arka plan ve ön plan tahminlerini tutan NumPy dizileri.
bgdmodel = np.zeros((1, 65), np.float64)
fgdmodel = np.zeros((1, 65), np.float64)

# GrabCut algoritmasını uygula
cv.grabCut(src, mask, r, bgdmodel, fgdmodel, 5, cv.GC_INIT_WITH_RECT)

# Maske işlemine göre sonucu oluştur
#Maske üzerindeki 0 ve 2 (arka plan) değerleri 0 yapılır.
#Diğer tüm değerler 1 yapılır.
#Bu işlem sonucunda mask2 adında yeni bir maske oluşturulur.
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
result = cv.bitwise_and(src, src, mask=mask2)

# Sonuçları göster
cv.imshow("Original Image with ROI", img)
cv.imshow("Segmented Result", result)
cv.imshow("Selected ROI", roi)
cv.waitKey(0)
cv.destroyAllWindows()
"""

#Çekirdek (Kernel)
#SVM algoritmasında çekirdek fonksiyonu, verilerin farklı boyutlara taşınarak ayrılabilir hale gelmesini sağlar.
    #OpenCV'de yaygın kullanılan çekirdek fonksiyonları:
        #cv.ml.SVM_LINEAR: Lineer çekirdek.
        #cv.ml.SVM_RBF: Radyal Baz Fonksiyonu çekirdeği (non-linear).
        #cv.ml.SVM_POLY: Polinom çekirdeği.
        #cv.ml.SVM_SIGMOID: Sigmoid çekirdeği.
    #setKernel() işlevi, önceden tanımlanmış bir çekirdek fonksiyonunu (kernel) SVM modeline belirtmek için kullanılır.
    #cv.ml.SVM_LINEAR, lineer çekirdeği temsil eder ve modelin lineer ayrılabilir veri üzerinde çalışmasını sağlar.

"""
#SVM KERNEL KULLANARAK SINIFLANDIRMA ÖRNEĞİ 
#BU KOD: İki sınıf veri kümesini (class1 ve class2) oluşturur ve SVM algoritması kullanarak sınıflandırır.
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Veri oluştur
np.random.seed(0)
#Seed kullanılarak her çalıştırmada aynı rastgele sayıların üretilmesi sağlanır.
class1 = np.random.randn(100, 2) + np.array([2, 2])  # Sınıf 1

#np.random.randn(100, 2):
    #100 satır ve 2 sütundan oluşan bir normal dağılım (Gaussian distribution) ile rastgele sayılar üretir.
        #Bu sayılar, ortalama = 0 ve standart sapma = 1 olan bir normal dağılımdan çekilir.
        #Her satır, 2 boyutlu bir veri noktasıdır.
    #np.array([2, 2]):
        #Üretilen her bir veri noktasına (2, 2) eklenir.
        #Bu, tüm veri noktalarını (2, 2) koordinatına taşır.
        #Yani, veri noktaları artık (2, 2) çevresinde yoğunlaşır.
    #Amaç:
        #class1 veri kümesi, 2 boyutlu bir düzlemde (2, 2) merkezi etrafında kümelenmiş bir veri grubudur.
class2 = np.random.randn(100, 2) + np.array([-2, -2])  # Sınıf 2
     #Amaç:
        #class2 veri kümesi, 2 boyutlu bir düzlemde (-2, -2) merkezi etrafında kümelenmiş bir veri grubudur.
        #Bu, class1'den farklı bir bölgede konumlanan başka bir veri kümesini temsil eder.
X = np.vstack((class1, class2)).astype(np.float32)  # Özellikler
    #np.vstack(): Veri kümesini dikey olarak birleştirir.
    #OpenCV'de makine öğrenimi algoritmaları, girdilerin veri tipinin float32 (32-bit float) olmasını bekler.
y = np.hstack((np.ones(100), -1 * np.ones(100)))    # Etiketler
    #p.ones(100):
        #100 adet 1'den oluşan bir vektör oluşturur.
        #Bu,class1'i temsil eden etiket kümesidir.
        #Yani, class1 veri kümesindeki her satıra 1 etiketi atanır.
    #-1 * np.ones(100):
        #100 adet -1'den oluşan bir vektör oluşturur.
        #Bu, class2'yi temsil eden etiket kümesidir.
        #Yani,class2 veri kümesindeki her satıra -1 etiketi atanır.
    #np.hstack((...)):
        #class1 etiketlerini ve class2 etiketlerini yatay olarak (horizontal stack) birleştirir.
        #Bu, 200 satırlık bir etiket dizisi oluşturur.
        #İlk 100 eleman 1 (class1), sonraki 100 eleman ise -1 (class2) olur.
    #Amaç:
        #Bu,X matrisindeki her bir veri noktasına karşılık gelen sınıf etiketlerini (y) oluşturur.
        #Sınıf 1 verileri için etiket = 1, sınıf 2 verileri için etiket = -1 olarak atanır.
print("X (özellik matrisi) boyutu:", X.shape)  # (200, 2)
print("X (özellik matrisi) veri tipi:", X.dtype)  # float32
print("y (etiket vektörü) boyutu:", y.shape)  # (200,)
print("y (etiket vektörü) ilk 10 değer:", y[:10])  # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
y = y.astype(np.int32)

# SVM modeli oluştur ve RBF kernel seç
svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_RBF)  # Kernel olarak RBF seçildi
svm.setType(cv.ml.SVM_C_SVC)  # Tür: C-Support Vector Classification
svm.setC(2.5)                 # C düzenleme parametresi
svm.setGamma(0.5)             # RBF için Gamma parametresi
svm.train(X, cv.ml.ROW_SAMPLE, y)

# Karar sınırını çiz
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #Amaç: Karar sınırını çizeceğimiz 2D düzlemin sınırlarını belirlemek.
    #X[:, 0]: Tüm veri noktalarının x koordinatları (1. sütun) alınır.
    #X[:, 1]: Tüm veri noktalarının y koordinatları (2. sütun) alınır.
    #X[:, 0].min() ve X[:, 0].max(): X ekseni için minimum ve maksimum değerleri bulur.
    #X[:, 1].min() ve X[:, 1].max(): Y ekseni için minimum ve maksimum değerleri bulur.
    #-1 ve +1 eklenir, böylece grafik sınırları biraz genişletilir. Bu, veri noktalarının sınırına ek boşluk ekler.


xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
    #Amaç: 2 boyutlu bir grid (ızgara) oluşturmak.
    #np.arange(x_min, x_max, 0.01):
    #x_min ile x_max arasındaki tüm değerleri 0.01 aralıklarla oluşturur.
    #Bu, x eksenindeki tüm olası koordinatları verir.
    #np.arange(y_min, y_max, 0.01):
    #Y eksenindeki tüm olası koordinatları 0.01 aralıklarla oluşturur.
    #np.meshgrid():
        #Bu iki diziyi (x ve y) birleştirir ve 2 boyutlu bir grid (ızgara) oluşturur.
        #xx: X ekseni için tekrarlanan değerleri içerir.
        #yy: Y ekseni için tekrarlanan değerleri içerir.

#Genel Akış
    #Koordinat Aralığı Belirle:
    #(x_min, x_max) ve (y_min, y_max) arasında bir alan tanımla.
    #2D Grid (Izgara) Oluştur:
    #xx ve yy ile 2D bir grid oluştur.
    #Grid'i Vektöre Dönüştür:
    #Tüm x, y koordinatlarını tek bir (x, y) vektörü halinde birleştir.
grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    #Amaç: 2D matrisleri 1D listeye (vektör) dönüştürmek.
    #xx.ravel(): 2D matris xx'yi 1D vektöre dönüştürür.
    #yy.ravel(): 2D matris yy'yi 1D vektöre dönüştürür.
    #np.c_[]:
        #xx.ravel() ve yy.ravel() vektörlerini birleştirir ve her satır bir (x, y) koordinat çifti oluşturur.
print(grid_points)
_, Z = svm.predict(grid_points)
Z = Z.reshape(xx.shape)
    #_, Z:
        #predict() fonksiyonu, iki değer döndürür: çıkış kodu ve sınıf tahminleri.
        #_: Bu, kullanılmayan bir değerdir, genellikle hata kodları için kullanılır.
        #Z: Bu, tahmin edilen sınıflardır (örneğin, -1 veya 1).

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.title("SVM with RBF Kernel")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
"""

#CAFFE İLE NESNE TESPİTİ
    #Caffe, derin öğrenme modelleri oluşturmak ve eğitmek için kullanılan popüler bir açık kaynaklı derin öğrenme kütüphanesidir.
    #Caffe, Convolutional Neural Networks (CNN) ve diğer derin öğrenme modelleri için birçok önceden eğitilmiş model sunar.
"""
#CAFFE İLE NESNE TESPİTİ
import cv2
import numpy as np

# Model ve ağırlık dosyalarının yolları
config_path = "path_to_model/deploy.prototxt"  # Konfigürasyon dosyası
weights_path = "path_to_model/res10_300x300_ssd_iter_140000_fp16.caffemodel"  # Eğitim ağırlıkları dosyası

# Önceden eğitilmiş modeli yükleme
net = cv2.dnn.readNetFromCaffe(config_path, weights_path)

# Kamera açma (web kamerayı başlatma)
cap = cv2.VideoCapture(0)  # 0, varsayılan kamera için

# Nesne isimleri (örnek bir liste, kendi modelinizin desteklediği sınıflara göre güncelleyebilirsiniz)
objName = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

while True:
    ret, frame = cap.read()

    if ret is False:
        break

    # Görüntü boyutlarını al
    h, w = frame.shape[:2]

    # Görüntüyü blob formatına dönüştür
    blobImage = cv2.dnn.blobFromImage(
        frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), True, False
    )

    # Blob'u modele giriş olarak ayarla
    net.setInput(blobImage)
    #net: Daha önce yüklenmiş sinir ağı modeli (önceden tanımlı olmalı).

    # Modelden çıktı al
    cvOut = net.forward()
    #Amaç: Modelin çıktılarını hesaplamak
    #net.forward(): Girdi görüntüsü üzerinden tahmin yapar ve tüm tespitleri döner.

    # Çıktıları işle
    for detection in cvOut[0, 0, :, :]:
        score = float(detection[2])  # Tespit güven skoru
        # cvOut[0, 0, :, :]: Tüm tespit edilen nesneleri döner.
        # detection[2]: Modelin tahmin ettiği nesnenin doğruluk oranı (confidence score).
        if score > 0.5:  # Güven skoru 0.5'ten büyükse
            objIndex = int(detection[1])  # Nesne sınıfı


            # Tespit edilen nesnenin koordinatlarını al
            left = detection[3] * w
            top = detection[4] * h
            right = detection[5] * w
            bottom = detection[6] * h

            # Tespit edilen nesnenin etrafına bir dikdörtgen çiz
            cv2.rectangle(frame,
                          (int(left), int(top)),
                          (int(right), int(bottom)),
                          (255, 0, 0), thickness=2)

            # Nesne adı ve skoru yazdır
            cv2.putText(
                frame,
                "score:%.2f, %s" % (score, objName[objIndex]),
                (int(left) - 10, int(top) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                8
            )

    # Sonuç görüntüsünü göster
    cv2.imshow("video-ssd-demo", frame)

    # ESC tuşu ile çıkış kontrolü
    c = cv2.waitKey(10)
    if c == 27:  # ESC'nin ASCII kodu
        break

# Kamera ve pencereleri serbest bırak
cap.release()
cv2.destroyAllWindows()
"""

#Temel Morfolojik İşlemler
    #Erozyon: Görüntüdeki nesneleri küçültmek için kullanılır.
    #Genişletme: Görüntüdeki nesneleri büyütmek için kullanılır.
    #Açma: Erozyon ve genişletme işlemlerini bir arada kullanarak nesneleri temizlemek için kullanılır.
    #Kapama: Genişletme ve erozyon işlemlerini bir arada kullanarak nesneleri doldurmak için kullanılır.
    #Gradyan: Genişletme ve erozyon işlemleri arasındaki farkı hesaplamak için kullanılır.
    #Top Hat: Görüntü ve açma işlemi sonucu oluşan farkı hesaplamak için kullanılır.
    #Black Hat: Görüntü ve kapama işlemi sonucu oluşan farkı hesaplamak için kullanılır.

########################################################################################################################################
#COMMAND + F ILE HIZLI ARAMA
#Creator: Niyazi Mert Işıksal
#Linkedin: https://www.linkedin.com/in/niyazi-mert-isiksal-8b7920281/


