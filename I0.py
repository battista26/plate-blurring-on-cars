# prompt: I.png ve I0.png dosyalarını gösteren kodu yaz matplotlib plt kullan

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import cv2

# Read the images using OpenCV
img1 = cv2.imread('I.png')
img2 = cv2.imread('I0.png')

# Convert images from BGR to RGB for matplotlib
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Create a figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Display the images
axes[0].imshow(img1_rgb)
axes[0].set_title('I.png')
axes[0].axis('off') # Hide axes ticks

axes[1].imshow(img2_rgb)
axes[1].set_title('I0.png')
axes[1].axis('off') # Hide axes ticks

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# prompt:  tuz biberi medyanla götüren fonksiyonu yaz

import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib import pyplot as plt

def median_filter_salt_and_pepper(image, kernel_size):
  """
  Applies a median filter to reduce salt and pepper noise in an image.

  Args:
    image: The input image (numpy array).
    kernel_size: The size of the median filter kernel (should be an odd integer).

  Returns:
    The image after applying the median filter.
  """
  if kernel_size % 2 == 0:
    raise ValueError("Kernel size must be an odd integer.")

  # Apply median filter
  filtered_image = cv2.medianBlur(image, kernel_size)
  return filtered_image

# Assuming img2 (I0.png) is the image with salt and pepper noise
# You can replace this with any image you want to process
noisy_image = img1

# Apply the median filter with a kernel size of 3 (you can adjust this)
filtered_image = median_filter_salt_and_pepper(noisy_image, 3)

# Convert filtered image from BGR to RGB for matplotlib
filtered_image_rgb = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)

# Display the original noisy image and the filtered image
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img1_rgb) # Display the original noisy image (I0.png)
axes[0].set_title('Original Noisy Image (I0.png)')
axes[0].axis('off')

axes[1].imshow(filtered_image_rgb)
axes[1].set_title('Median Filtered Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# 1.2 – Gri tonlama ve kontrast artırma (img2 üzerinden)
# Gri tona çevir
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 1) Global histogram eşitleme
equalized2 = cv2.equalizeHist(gray2)

# 2) Alternatif – CLAHE (Contrast Limited AHE)
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(11,11))
enhanced2 = clahe.apply(gray2)

# Sonuçları matplotlib ile göstermek için
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
ax1.imshow(gray2,      cmap='gray'); ax1.set_title('Grayscale');      ax1.axis('off')
ax2.imshow(equalized2, cmap='gray'); ax2.set_title('Equalized');     ax2.axis('off')
ax3.imshow(enhanced2,  cmap='gray'); ax3.set_title('CLAHE Enhanced'); ax3.axis('off')
plt.tight_layout()
plt.show()

cv2.imwrite('Clahe.png',enhanced2)
cv2.imwrite('equalize.png',equalized2)
cv2.imwrite('gray.png',gray2)


# 1.3 – Gürültü giderme: CLAHE sonrası görüntüye Bilateral Filter uygulama
# enhanced2 değişkeni, daha önce CLAHE ile kontrastı artırılmış gri tonlu görüntü

# Parametreler: d=9 (komşuluk yarıçapı), sigmaColor=75, sigmaSpace=75
bilateral2 = cv2.bilateralFilter(enhanced2, d=5, sigmaColor=50, sigmaSpace=50)

# Sonuçları karşılaştırmalı olarak göster
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.imshow(enhanced2, cmap='gray')
ax1.set_title('CLAHE Enhanced')
ax1.axis('off')

ax2.imshow(bilateral2, cmap='gray')
ax2.set_title('Bilateral Filtered')
ax2.axis('off')

plt.tight_layout()
plt.show()
cv2.imwrite('bilat.png',bilateral2)


# 2. Kenar tespiti ve morfolojik kapama (bilateral filtreden sonra)

# Canny ile kenar tespiti (eşikler: 50–150 arası deneyebilirsin)
fig, (ax1, ax2 ) = plt.subplots(1, 2, figsize=(12, 5))

edges2 = cv2.Canny(bilateral2, 120, 260)
ax1.imshow(edges2, cmap='gray')
ax1.set_title('Canny Kenarları')
ax1.axis('off')

# Morfolojik kapanış: küçük boşlukları kapatmak için yatay-dikey bir kernel
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20,3))# 3 en iyi

closed2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, kernel2)
ax2.imshow(closed2, cmap='gray')
ax2.set_title('Morfolojik Kapanış')
ax2.axis('off')


# Sonuçları göster

plt.tight_layout()
plt.show()

# 2.3 – Erosion uygulama (morfolojik kapanış sonrası)
# closed2 değişkeni, önceki morfolojik kapanış sonucunu içeriyor

# Erosion için küçük bir dikdörtgen çekirdek
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 4))

# Erode işlemi (iterations ile erozyon kuvvetini artırabilirsin)
eroded2 = cv2.erode(closed2, kernel3, iterations=2)

# Sonucu göster
plt.figure(figsize=(6, 5))
plt.imshow(eroded2, cmap='gray')
plt.title('Eroded (Aşındırılmış) Kenarlar')
plt.axis('off')
plt.show()

cv2.imwrite('eroded.png',eroded2)



# 2.4 – Alan büyüklüğüne göre bileşen filtresi
min_area = 150   # en küçük piksel sayısı eşiği
max_area = 1700   # en büyük piksel sayısı eşiği

# connectedComponentsWithStats ile bileşenleri bul
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded2, connectivity=8)

# Yeni maskeyi oluştur
filtered_mask = np.zeros_like(eroded2)

# Her bileşeni kontrol et
for label in range(1, num_labels):
    area = stats[label, cv2.CC_STAT_AREA]
    if min_area <= area <= max_area:
        filtered_mask[labels == label] = 255

# Sonucu göster
plt.figure(figsize=(6,6))
plt.imshow(filtered_mask, cmap='gray')
plt.title(f'Alanı {min_area}-{max_area} piksel arası olan bileşenler')
plt.axis('off')
plt.show()

import numpy as np

# 2.5 – Boy/En oranı (h/w) 1/7 ile 2/7 arasında olan bileşenleri filtreleme
min_hw = 1/7.0
max_hw = 2/7.0

# filtered_mask, önceki alan filtresinden gelen ikili maske
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filtered_mask, connectivity=8)

# Yeni maske
hw_mask = np.zeros_like(filtered_mask)

for label in range(1, num_labels):
    w = stats[label, cv2.CC_STAT_WIDTH]
    h = stats[label, cv2.CC_STAT_HEIGHT]
    hw = h / float(w)
    if min_hw <= hw <= max_hw:
        hw_mask[labels == label] = 255

# Sonucu göster
plt.figure(figsize=(6,6))
plt.imshow(hw_mask, cmap='gray')
plt.title(f'Boy/En oranı ∈ [{min_hw:.3f}, {max_hw:.3f}] olan bileşenler')
plt.axis('off')
plt.show()

# Erosion için küçük bir dikdörtgen çekirdek
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

# Erode işlemi (iterations ile erozyon kuvvetini artırabilirsin)
eroded_last = cv2.erode(hw_mask, kernel3, iterations=1)

# Sonucu göster
plt.figure(figsize=(6, 5))
plt.imshow(eroded_last, cmap='gray')
plt.title('Eroded (Aşındırılmış) Kenarlar')
plt.axis('off')
plt.show()


# 2.5 – Boy/En oranı (h/w) 1/7 ile 2/7 arasında olan bileşenleri filtreleme
min_hw = 0.5/7.0
max_hw = 2/7.0

# filtered_mask, önceki alan filtresinden gelen ikili maske
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded_last, connectivity=8)

# Yeni maske
hw_mask = np.zeros_like(eroded_last)

for label in range(1, num_labels):
    w = stats[label, cv2.CC_STAT_WIDTH]
    h = stats[label, cv2.CC_STAT_HEIGHT]
    hw = h / float(w)
    if min_hw <= hw <= max_hw:
        hw_mask[labels == label] = 255

# Sonucu göster
plt.figure(figsize=(6,6))
plt.imshow(hw_mask, cmap='gray')
plt.title(f'Boy/En oranı ∈ [{min_hw:.3f}, {max_hw:.3f}] olan bileşenler')
plt.axis('off')
plt.show()

# 2.6 – Dilation uygulama (hw_mask veya istediğin ikili maske üzerinde)

# 3×3 dikdörtgen çekirdek tanımla
kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 7))

# Dilate işlemi; iterations ile büyütme kuvvetini ayarlayabilirsin
dilated2 = cv2.dilate(hw_mask, kernel4, iterations=2)

# Sonucu göster
plt.figure(figsize=(6,6))
plt.imshow(dilated2, cmap='gray')
plt.title('Dilate Uygulanmış Maske')
plt.axis('off')
plt.show()

cv2.imwrite('dilated.png',dilated2)

# 2.7 – Orijinal görüntüden dilated2 maskesini çıkarma (karartma)
# img2: orijinal BGR görüntü
# dilated2: 0/255 ikili maske

# 1) Maskeyi ters çevir (beyaz alanlar çıkarılacak bölge)
inv_mask = cv2.bitwise_not(dilated2)

# 2) Orijinal görüntüye tersi maskeyi uygula
#    Böylece maskelenen bölge siyaha boyanır
result = cv2.bitwise_and(img2, img2, mask=inv_mask)

# 3) Sonucu RGB’ye çevirip göster
plt.figure(figsize=(8,5))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Maskelenen Bölgeler Çıkarılmış Hali')
plt.show()

result = img2.copy()
contours, _ = cv2.findContours(dilated2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    roi     = result[y:y+h, x:x+w]
    blurred = cv2.GaussianBlur(roi, (25,25), 0)
    result[y:y+h, x:x+w] = blurred

plt.figure(figsize=(8,5))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('ROI Bazlı Gaussian Blur')
plt.show()

# 1) Tüm görüntüyü bulanıklaştır
blurred_full = cv2.GaussianBlur(img2, (25,25), 0)

# 2) Kopya oluştur
result = img2.copy()

# 3) Sadece maskelenen bölgeyi blurred_full'dan result'a kopyala
#    (mask'in tek kanallı ve 0/255 değerlerinde olduğundan emin ol)
cv2.copyTo(blurred_full, dilated2, result)

# 4) Göster
plt.figure(figsize=(8,5))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('copyTo ile Sadece Dilate Bölgeleri Blurlandı')
plt.show()