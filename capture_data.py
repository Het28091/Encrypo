import cv2
import os

cap = cv2.VideoCapture(0)

img_counter = 0

fileName = input("Enter you name in 'FIRSTNAME_LASTNAME' format")
save_dir = f"captured_images_of_{fileName}"

os.makedirs(save_dir)

limit = 100

while cap.isOpened() and img_counter < limit:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("image", frame)

    img_name = os.path.join(save_dir, f"{fileName}_{img_counter+1}.png")
    cv2.imwrite(img_name, frame)
    img_counter += 1

cap.release()
cv2.destroyAllWindows()

print(f"Saved {img_counter} photos in {save_dir}")
