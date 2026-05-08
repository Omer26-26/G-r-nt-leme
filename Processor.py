import numpy as np
import matplotlib.pyplot as plt


class ImageProcessor:

    def __init__(self):
        self.image = None

    @staticmethod
    def turn_gray(image):
        R = image[:, :, 0]
        G = image[:, :, 1]
        B = image[:, :, 2]

        gray = R * 0.299 + G * 0.587 + B * 0.114
        return gray.astype(np.uint8)

    @staticmethod
    def turn_binary(image, threshold=127):
        if image.ndim == 3:
            image = ImageProcessor.turn_gray(image)

        binary = (image > threshold).astype(np.uint8) * 255
        return binary

    @staticmethod
    def adaptive_threshold_manual(image, block_size=15, C=5):
        # Mali - Adaptif eşikleme manuel olarak piksel komşuluk ortalamasıyla uygulanır.
        if block_size % 2 == 0:
            block_size += 1

        if image.ndim == 3:
            gray = ImageProcessor.turn_gray(image)
        else:
            gray = image.copy()

        gray = gray.astype(np.float64)
        height, width = gray.shape
        pad = block_size // 2
        padded = np.pad(
            gray,
            ((pad, pad), (pad, pad)),
            mode="constant",
            constant_values=0,
        )
        output = np.zeros((height, width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                total = 0.0
                for ki in range(block_size):
                    for kj in range(block_size):
                        total += padded[i + ki, j + kj]
                local_mean = total / (block_size * block_size)
                threshold = local_mean - C
                output[i, j] = 255 if gray[i, j] > threshold else 0

        return output

    @staticmethod
    def sobel_edge_manual(image, threshold=None):
        # Mali - Sobel kenar bulma manuel olarak Gx ve Gy maskeleriyle uygulanır.
        if image.ndim == 3:
            gray = ImageProcessor.turn_gray(image)
        else:
            gray = image.copy()

        gray = gray.astype(np.float64)
        height, width = gray.shape

        gx_kernel = np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ],
            dtype=np.float64,
        )
        gy_kernel = np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1],
            ],
            dtype=np.float64,
        )

        padded = np.pad(gray, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        magnitude = np.zeros((height, width), dtype=np.float64)

        for i in range(height):
            for j in range(width):
                gx = 0.0
                gy = 0.0

                for ki in range(3):
                    for kj in range(3):
                        pixel = padded[i + ki, j + kj]
                        gx += pixel * gx_kernel[ki, kj]
                        gy += pixel * gy_kernel[ki, kj]

                magnitude[i, j] = np.sqrt(gx * gx + gy * gy)

        max_value = np.max(magnitude)
        if max_value > 0:
            normalized = magnitude * (255.0 / max_value)
        else:
            normalized = magnitude

        if threshold is not None:
            return (normalized >= threshold).astype(np.uint8) * 255

        return normalized.astype(np.uint8)

    @staticmethod
    def add_salt_pepper_noise_manual(image, amount=0.05, seed=None):
        # Mali - Salt & Pepper gürültüsü manuel olarak rastgele pikseller 0 veya 255 yapılarak eklenir.
        amount = max(0.0, min(1.0, amount))
        noisy = image.copy()
        rng = np.random.default_rng(seed)

        height = noisy.shape[0]
        width = noisy.shape[1]
        pepper_limit = amount / 2
        salt_limit = 1 - (amount / 2)

        for i in range(height):
            for j in range(width):
                random_value = rng.random()

                if random_value < pepper_limit:
                    if noisy.ndim == 3:
                        noisy[i, j] = [0, 0, 0]
                    else:
                        noisy[i, j] = 0
                elif random_value > salt_limit:
                    if noisy.ndim == 3:
                        noisy[i, j] = [255, 255, 255]
                    else:
                        noisy[i, j] = 255

        return noisy.astype(np.uint8)

    @staticmethod
    def mean_filter_manual(image, kernel_size=3):
        # Mali - Mean filtre manuel olarak komşuluk penceresindeki piksel ortalamasıyla uygulanır.
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1

        pad = kernel_size // 2
        source = image.copy().astype(np.float64)
        height = source.shape[0]
        width = source.shape[1]

        result = np.zeros_like(source, dtype=np.float64)

        if source.ndim == 3:
            channel_count = source.shape[2]
            padded = np.pad(
                source,
                ((pad, pad), (pad, pad), (0, 0)),
                mode="edge",
            )

            for i in range(height):
                for j in range(width):
                    for channel in range(channel_count):
                        total = 0.0
                        for ki in range(kernel_size):
                            for kj in range(kernel_size):
                                total += padded[i + ki, j + kj, channel]
                        result[i, j, channel] = total / (kernel_size * kernel_size)
        else:
            padded = np.pad(
                source,
                ((pad, pad), (pad, pad)),
                mode="edge",
            )

            for i in range(height):
                for j in range(width):
                    total = 0.0
                    for ki in range(kernel_size):
                        for kj in range(kernel_size):
                            total += padded[i + ki, j + kj]
                    result[i, j] = total / (kernel_size * kernel_size)

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def turn_blur(image, kernel_size=3):
        return ImageProcessor.mean_filter_manual(image, kernel_size)

    @staticmethod
    def median_filter_manual(image, kernel_size=3):
        # Mali - Median filtre manuel olarak komşuluk değerleri sıralanıp ortanca değer alınarak uygulanır.
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1

        pad = kernel_size // 2
        source = image.copy()
        height = source.shape[0]
        width = source.shape[1]
        result = np.zeros_like(source, dtype=np.float64)

        if source.ndim == 3:
            channel_count = source.shape[2]
            padded = np.pad(
                source,
                ((pad, pad), (pad, pad), (0, 0)),
                mode="edge",
            )

            for i in range(height):
                for j in range(width):
                    for channel in range(channel_count):
                        values = []
                        for ki in range(kernel_size):
                            for kj in range(kernel_size):
                                values.append(padded[i + ki, j + kj, channel])

                        for sort_i in range(len(values) - 1):
                            min_index = sort_i
                            for sort_j in range(sort_i + 1, len(values)):
                                if values[sort_j] < values[min_index]:
                                    min_index = sort_j
                            temp = values[sort_i]
                            values[sort_i] = values[min_index]
                            values[min_index] = temp

                        middle_index = len(values) // 2
                        result[i, j, channel] = values[middle_index]
        else:
            padded = np.pad(
                source,
                ((pad, pad), (pad, pad)),
                mode="edge",
            )

            for i in range(height):
                for j in range(width):
                    values = []
                    for ki in range(kernel_size):
                        for kj in range(kernel_size):
                            values.append(padded[i + ki, j + kj])

                    for sort_i in range(len(values) - 1):
                        min_index = sort_i
                        for sort_j in range(sort_i + 1, len(values)):
                            if values[sort_j] < values[min_index]:
                                min_index = sort_j
                        temp = values[sort_i]
                        values[sort_i] = values[min_index]
                        values[min_index] = temp

                    middle_index = len(values) // 2
                    result[i, j] = values[middle_index]

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _to_binary_if_needed(image):
        if image.ndim == 3:
            return ImageProcessor.turn_binary(image)
        return image

    @staticmethod
    def turn_dilate(image, kernel_size=3):
        image = ImageProcessor._to_binary_if_needed(image)

        height = image.shape[0]
        width = image.shape[1]
        pad = kernel_size // 2

        output = np.zeros_like(image)
        padded = np.pad(
            image,
            ((pad, pad), (pad, pad)),
            mode="constant",
            constant_values=0,
        )

        for i in range(height):
            for j in range(width):
                window = padded[i:i + kernel_size, j:j + kernel_size]
                output[i, j] = np.max(window)

        return output.astype(np.uint8)

    @staticmethod
    def turn_erode(image, kernel_size=3):
        image = ImageProcessor._to_binary_if_needed(image)

        height = image.shape[0]
        width = image.shape[1]
        pad = kernel_size // 2

        output = np.zeros_like(image)
        padded = np.pad(
            image,
            ((pad, pad), (pad, pad)),
            mode="constant",
            constant_values=255,
        )

        for i in range(height):
            for j in range(width):
                window = padded[i:i + kernel_size, j:j + kernel_size]
                output[i, j] = np.min(window)

        return output.astype(np.uint8)

    @staticmethod
    def turn_opening(image, kernel_size=3):
        eroded = ImageProcessor.turn_erode(image, kernel_size)
        return ImageProcessor.turn_dilate(eroded, kernel_size)

    @staticmethod
    def turn_closing(image, kernel_size=3):
        dilated = ImageProcessor.turn_dilate(image, kernel_size)
        return ImageProcessor.turn_erode(dilated, kernel_size)

    @staticmethod
    def stretch_histogram_manual(image):
        if image.ndim == 3:
            img_work = ImageProcessor.turn_gray(image)
        else:
            img_work = image.copy()

        img_min = np.min(img_work)
        img_max = np.max(img_work)

        if img_max == img_min:
            return img_work

        stretched = (img_work - img_min) * (255.0 / (img_max - img_min))
        return stretched.astype(np.uint8)

    @staticmethod
    def rgb_to_hsv_manual(image):
        img = image.astype(np.float32) / 255.0
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        v = np.max(img, axis=2)
        m = np.min(img, axis=2)
        diff = v - m

        s = np.zeros_like(v)
        s[v != 0] = diff[v != 0] / v[v != 0]

        h = np.zeros_like(v)

        idx = (v == r) & (diff != 0)
        h[idx] = (60 * ((g[idx] - b[idx]) / diff[idx]) + 360) % 360

        idx = (v == g) & (diff != 0)
        h[idx] = (60 * ((b[idx] - r[idx]) / diff[idx]) + 120) % 360

        idx = (v == b) & (diff != 0)
        h[idx] = (60 * ((r[idx] - g[idx]) / diff[idx]) + 240) % 360

        h_final = (h / 2).astype(np.uint8)
        s_final = (s * 255).astype(np.uint8)
        v_final = (v * 255).astype(np.uint8)

        return np.stack([h_final, s_final, v_final], axis=2)

    @staticmethod
    def resize_manual(image, scale_factor):
        old_h, old_w = image.shape[:2]
        new_h = int(old_h * scale_factor)
        new_w = int(old_w * scale_factor)

        row_indices = (np.arange(new_h) / scale_factor).astype(int)
        col_indices = (np.arange(new_w) / scale_factor).astype(int)

        row_indices = np.clip(row_indices, 0, old_h - 1)
        col_indices = np.clip(col_indices, 0, old_w - 1)

        if image.ndim == 3:
            return image[np.ix_(row_indices, col_indices, [0, 1, 2])]
        return image[np.ix_(row_indices, col_indices)]

    @staticmethod
    def get_histogram(image):
        if image.ndim == 3:
            image = ImageProcessor.turn_gray(image)

        hist = np.zeros(256, dtype=int)
        flat_image = image.ravel()

        for pixel in flat_image:
            hist[pixel] += 1

        return hist

    @staticmethod
    def plot_histogram(image, title="Histogram"):
        hist = ImageProcessor.get_histogram(image)
        plt.figure()
        plt.title(title)
        plt.bar(range(256), hist, color="gray")
        plt.show()
