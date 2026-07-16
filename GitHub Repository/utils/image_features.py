import cv2
import numpy as np


def detect_skin(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    hsv_mask = cv2.inRange(
        hsv,
        np.array([0, 20, 40]),
        np.array([35, 255, 255])
    )

    ycrcb_mask = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77]),
        np.array([255, 173, 127])
    )

    mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def save_skin_mask(image, filename):

    mask = detect_skin(image)

    overlay = image.copy()

    overlay[mask == 0] = (0, 0, 0)

    cv2.imwrite(filename, overlay)


def extract_features(image):

    mask = detect_skin(image)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    features = []

    for color_space in [rgb, hsv, lab]:

        for channel in cv2.split(color_space):

            pixels = channel[mask > 0]

            # Skip images where skin detection failed
            if len(pixels) < 100:
                return None

            pixels = pixels.astype(np.float32)

            median = np.median(pixels)

            distance = np.abs(pixels - median)

            threshold = np.percentile(distance, 70)

            pixels = pixels[distance <= threshold]

            features.extend([
                np.mean(pixels),
                np.median(pixels),
                np.std(pixels),
                np.percentile(pixels, 10),
                np.percentile(pixels, 25),
                np.percentile(pixels, 75),
                np.percentile(pixels, 90),
                np.min(pixels),
                np.max(pixels)
            ])

    return features