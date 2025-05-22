import cv2
import matplotlib.pyplot as plt
import numpy as np

def median_filter_salt_and_pepper(image, kernel_size):
  if kernel_size % 2 == 0:
    raise ValueError("Kernel size must be an odd integer.")

  # Apply median filter
  filtered_image = cv2.medianBlur(image, kernel_size)
  return filtered_image

# Görüntüyü yükle
original_img = cv2.imread('I.png')
if original_img is None:
    raise FileNotFoundError("'I.png' yüklenemedi.")
original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

original_img = median_filter_salt_and_pepper(original_img,5)
filtered_image_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)


print("Image shape:", original_img.shape, "dtype:", original_img.dtype)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original_img_rgb)
axes[0].set_title('Orijinal Fotoğraf')
axes[0].axis('off')

axes[1].imshow(filtered_image_rgb)
axes[1].set_title('Medyan Filtreli Fotoğraf')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Gri tonlama ve gürültü azaltma
gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

# Gauss
gauss = cv2.GaussianBlur(gray, (5, 5), 3)

# Canny kenar tespiti
edges = cv2.Canny(gauss, 40, 80)

# Canny çıktısını görselleştir
fig, axes = plt.subplots(1, 3, figsize=(10, 5))
axes[0].imshow(gray, cmap='gray')
axes[0].set_title('Gri Tonlama ve Gürültü Azaltma')
axes[0].axis('off')

axes[1].imshow(gauss, cmap='gray')
axes[1].set_title('Gauss')
axes[1].axis('off')

axes[2].imshow(edges, cmap='gray')
axes[2].set_title('Canny Kenar Tespiti')
axes[2].axis('off')

plt.tight_layout()
plt.show()

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Tespit edilen kontur sayısı:", len(contours))

plt.imshow(cv2.drawContours(gray, contours, -1, (0, 255, 0), 3), cmap="gray")
plt.title('Konturların Çizilmesi')
plt.axis('off')
plt.show()

# Plaka adaylarını filtrele
plate_candidates = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    area = cv2.contourArea(contour)
    print(f"Alan: {area}, En-Boy Oranı: {aspect_ratio}")

    if (area > 500 and area < 100000) and (aspect_ratio > 1.5 and aspect_ratio < 6):
        plate_candidates.append(contour)

        # Kontur için maske
        mask = np.zeros_like(original_img)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

        # Gauss bulanıklığı uygula
        blurred = cv2.GaussianBlur(original_img, (25, 25), 30)

        # Bulanıklaştırılmış alanı yalnızca maskenin beyaz olduğu yerlerde birleştir
        original_img = np.where(mask == 255, blurred, original_img)

mask_img = np.zeros_like(original_img)
cv2.drawContours(mask_img, plate_candidates, -1, (255, 255, 255), -1)

temp_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
blurred = cv2.GaussianBlur(temp_rgb, (25, 25), 30)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(mask_img)
axes[0].set_title('Maskenin Görünümü')
axes[0].axis('off')

axes[1].imshow(blurred)
axes[1].set_title('Bulanıklaştırılmış Görüntü')
axes[1].axis('off')

plt.tight_layout()
plt.show()

print("Plaka adayı sayısı:", len(plate_candidates))

# Sonuçları göster
original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
plt.imshow(original_img_rgb)
plt.title('Sonuç')
plt.axis('off')
plt.show()