import numpy as np
import cv2


# ------- fft tools ------- #
def fftd(img, backwards=False):
    return cv2.dft(
        np.float32(img),
        flags=((cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT),
    )


def real(img):
    return img[:, :, 0]


def complexMultiplication(a, b):
    res = np.zeros(a.shape, a.dtype)
    res[:, :, 0] = a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]
    res[:, :, 1] = a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]
    return res


def complexDivision(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1.0 / (b[:, :, 0] ** 2 + b[:, :, 1] ** 2 + 1e-12)
    res[:, :, 0] = (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1]) * divisor
    res[:, :, 1] = (a[:, :, 1] * b[:, :, 0] + a[:, :, 0] * b[:, :, 1]) * divisor
    return res


def rearrange(img):
    assert img.ndim == 2
    img_ = np.zeros(img.shape, img.dtype)
    xh, yh = img.shape[1] // 2, img.shape[0] // 2
    img_[0:yh, 0:xh], img_[yh:img.shape[0], xh:img.shape[1]] = img[yh:img.shape[0], xh:img.shape[1]], img[0:yh, 0:xh]
    img_[0:yh, xh:img.shape[1]], img_[yh:img.shape[0], 0:xh] = img[yh:img.shape[0], 0:xh], img[0:yh, xh:img.shape[1]]
    return img_


# ------- rectangle tools ------- #
def x2(rect):
    return rect[0] + rect[2]


def y2(rect):
    return rect[1] + rect[3]


def limit(rect, limit_rect):
    if rect[0] + rect[2] > limit_rect[0] + limit_rect[2]:
        rect[2] = limit_rect[0] + limit_rect[2] - rect[0]
    if rect[1] + rect[3] > limit_rect[1] + limit_rect[3]:
        rect[3] = limit_rect[1] + limit_rect[3] - rect[1]
    if rect[0] < limit_rect[0]:
        rect[2] -= limit_rect[0] - rect[0]
        rect[0] = limit_rect[0]
    if rect[1] < limit_rect[1]:
        rect[3] -= limit_rect[1] - rect[1]
        rect[1] = limit_rect[1]
    rect[2] = max(rect[2], 0)
    rect[3] = max(rect[3], 0)
    return rect


def getBorder(original, limited):
    return [limited[0] - original[0], limited[1] - original[1], x2(original) - x2(limited), y2(original) - y2(limited)]


def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
    cutWindow = list(window)
    limit(cutWindow, [0, 0, img.shape[1], img.shape[0]])
    assert cutWindow[2] > 0 and cutWindow[3] > 0
    border = getBorder(window, cutWindow)
    res = img[cutWindow[1]:cutWindow[1] + cutWindow[3], cutWindow[0]:cutWindow[0] + cutWindow[2]]
    if border != [0, 0, 0, 0]:
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
    return res


class BaseTracker:
    """
    保持 run.py 不变的相关滤波改良版：
    1. 仍然使用 padded ROI + 频域相关滤波
    2. 尺度不再“强推放大”，而是围绕当前尺度做候选搜索，谁分数最好选谁
    3. 保留抗收缩，但只做弱约束，避免最后追成车灯
    4. 低置信度时扩大搜索范围，但不直接硬把框放大到超过目标
    """

    def __init__(self, hog=False, fixed_window=True, multiscale=False):
        self._interp_factor = 0.04
        self._padding = 2.8
        self._sigma_factor = 0.115
        self._lambda = 1e-4
        self._tmpl_sz = np.array([64, 64], dtype=np.int32)

        self._roi = [0.0, 0.0, 0.0, 0.0]
        self._tmpl = None
        self._alphaf = None
        self._prob = None
        self._cos_window = None

        # 统一的尺度候选：不是偏向放大，而是当前尺度周围的一圈候选
        self._scale_pool = [0.93, 0.97, 1.00, 1.03, 1.05, 1.08, 1.12]
        self._rescue_scale_pool = [0.90, 0.95, 1.00, 1.05, 1.10, 1.16, 1.24]

        self._peak_threshold = 0.06
        self._psr_threshold = 5.8
        self._good_psr = 8.0

        self._frame_id = 0
        self._warmup_frames = 10
        self._velocity = np.zeros(2, dtype=np.float32)
        self._low_conf_streak = 0
        self._shrink_streak = 0

        self._init_size = None
        self._size_ref = None
        self._min_size_ratio = 0.80
        self._max_shrink_per_frame = 0.03
        self._max_grow_per_frame = 0.12
        self._scale_smooth = 0.65

    def _create_gaussian_peak(self, sizey, sizex):
        output_sigma = np.sqrt(sizex * sizey) / self._padding * self._sigma_factor
        syh, sxh = sizey / 2.0, sizex / 2.0
        y, x = np.ogrid[0:sizey, 0:sizex]
        y = (y - syh) ** 2
        x = (x - sxh) ** 2
        g = np.exp(-0.5 * (x + y) / (output_sigma * output_sigma + 1e-12))
        return fftd(g)

    def _roi_center(self, roi):
        return np.array([roi[0] + roi[2] / 2.0, roi[1] + roi[3] / 2.0], dtype=np.float32)

    def _make_roi_from_center(self, center, size):
        return [center[0] - size[0] / 2.0, center[1] - size[1] / 2.0, size[0], size[1]]

    def _get_padded_roi(self, roi, scale=1.0, center_override=None, extra_expand=1.0):
        center = self._roi_center(roi) if center_override is None else np.array(center_override, dtype=np.float32)
        w = roi[2] * (1.0 + self._padding) * scale * extra_expand
        h = roi[3] * (1.0 + self._padding) * scale * extra_expand
        return [center[0] - w / 2.0, center[1] - h / 2.0, w, h]

    def _compute_feature(self, z):
        if z.ndim == 3 and z.shape[2] == 3:
            gray = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
        else:
            gray = z
        gray = gray.astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        if np.max(grad) > 1e-6:
            grad = grad / (np.max(grad) + 1e-6)
        feat = 0.75 * (gray - 0.5) + 0.25 * (grad - np.mean(grad))
        return feat.astype(np.float32)

    def getFeatures(self, image, roi, needed_size):
        roi = [int(round(v)) for v in roi]
        z = subwindow(image, roi, cv2.BORDER_REPLICATE)
        if z.shape[1] != needed_size[0] or z.shape[0] != needed_size[1]:
            z = cv2.resize(z, tuple(needed_size))
        feat = self._compute_feature(z)
        if self._cos_window is not None and feat.shape == tuple(self._tmpl_sz[::-1]):
            feat = feat * self._cos_window
        return feat

    def gaussianCorrelation(self, x1, x2):
        c = cv2.mulSpectrums(fftd(x1), fftd(x2), 0, conjB=True)
        c = fftd(c, True)
        c = real(c)
        c = rearrange(c)
        d = (np.sum(x1 * x1) + np.sum(x2 * x2) - 2.0 * c) / (self._tmpl_sz[0] * self._tmpl_sz[1])
        d = np.maximum(d, 0)
        return np.exp(-d / (self._sigma_factor * self._sigma_factor))

    def _calc_psr(self, response, peak_loc, side_win=11):
        px, py = peak_loc
        h, w = response.shape[:2]
        x1 = max(0, px - side_win // 2)
        x2 = min(w, px + side_win // 2 + 1)
        y1 = max(0, py - side_win // 2)
        y2 = min(h, py + side_win // 2 + 1)
        sidelobe = response.copy()
        sidelobe[y1:y2, x1:x2] = 0.0
        vals = sidelobe.flatten()
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return 0.0
        return float((response[py, px] - np.mean(vals)) / (np.std(vals) + 1e-6))

    def _subpixel_peak(self, response, peak_loc):
        x, y = peak_loc
        h, w = response.shape[:2]

        def interp_1d(left, center, right):
            denom = left - 2.0 * center + right
            if abs(denom) < 1e-6:
                return 0.0
            return 0.5 * (left - right) / denom

        dx, dy = 0.0, 0.0
        if 1 <= x < w - 1:
            dx = interp_1d(response[y, x - 1], response[y, x], response[y, x + 1])
        if 1 <= y < h - 1:
            dy = interp_1d(response[y - 1, x], response[y, x], response[y + 1, x])
        return float(np.clip(dx, -0.5, 0.5)), float(np.clip(dy, -0.5, 0.5))

    def _candidate_pool(self):
        return self._rescue_scale_pool if self._low_conf_streak >= 2 else self._scale_pool

    def _scale_penalty(self, s, peak, psr):
        """
        给偏离 1.0 的尺度一个温和惩罚，避免“动不动就放大/缩小”。
        但如果响应和 PSR 明显更好，仍然能赢。
        """
        deviation = abs(s - 1.0)
        conf = min(max(psr, 0.0), 12.0)
        # 置信度高时惩罚稍微减小，允许真正需要的尺度变化通过
        return 0.020 * deviation * (1.2 - 0.04 * conf) * max(0.5, 1.0 - 0.8 * peak)

    def track(self, search_region, img):
        best = None
        search_center_x = search_region[0] + search_region[2] / 2.0
        search_center_y = search_region[1] + search_region[3] / 2.0

        for scale_factor in self._candidate_pool():
            scaled_search_region = [
                search_center_x - search_region[2] * scale_factor / 2.0,
                search_center_y - search_region[3] * scale_factor / 2.0,
                search_region[2] * scale_factor,
                search_region[3] * scale_factor,
            ]
            z = self.getFeatures(img, scaled_search_region, self._tmpl_sz)
            k = self.gaussianCorrelation(z, self._tmpl)
            response = real(fftd(complexMultiplication(self._alphaf, fftd(k)), True))
            _, peak_value, _, peak_loc = cv2.minMaxLoc(response)
            dx, dy = self._subpixel_peak(response, peak_loc)
            psr = self._calc_psr(response, peak_loc)
            px = peak_loc[0] + dx
            py = peak_loc[1] + dy

            score = peak_value + 0.012 * min(psr, 12.0) - self._scale_penalty(scale_factor, peak_value, psr)
            item = (score, peak_value, psr, px, py, scale_factor)
            if best is None or item[0] > best[0] or (abs(item[0] - best[0]) < 1e-6 and psr > best[2]):
                best = item

        _, best_peak, best_psr, best_x, best_y, best_scale = best
        return best_x, best_y, best_scale, best_peak, best_psr

    def update_model(self, x, train_interp_factor):
        x = x.astype(np.float32)
        k = self.gaussianCorrelation(x, x)
        alphaf = complexDivision(self._prob, fftd(k) + self._lambda)
        if self._tmpl is None:
            self._tmpl = x
            self._alphaf = alphaf
            return
        self._tmpl = (1.0 - train_interp_factor) * self._tmpl + train_interp_factor * x
        self._alphaf = (1.0 - train_interp_factor) * self._alphaf + train_interp_factor * alphaf

    def init(self, roi, image):
        self._roi = list(map(float, roi))
        self._cos_window = np.outer(np.hanning(self._tmpl_sz[1]), np.hanning(self._tmpl_sz[0])).astype(np.float32)
        self._prob = self._create_gaussian_peak(self._tmpl_sz[1], self._tmpl_sz[0])
        x = self.getFeatures(image, self._get_padded_roi(self._roi, scale=1.0), self._tmpl_sz)
        self.update_model(x, 1.0)
        size = np.array([self._roi[2], self._roi[3]], dtype=np.float32)
        self._init_size = size.copy()
        self._size_ref = size.copy()
        self._velocity[:] = 0.0
        self._frame_id = 0
        self._low_conf_streak = 0
        self._shrink_streak = 0

    def update(self, image):
        self._frame_id += 1
        h_img, w_img = image.shape[:2]

        current_center = self._roi_center(self._roi)
        predicted_center = current_center + self._velocity
        speed = float(np.linalg.norm(self._velocity))

        # 只扩大搜索范围，不直接硬放大框
        extra_expand = 1.0
        if self._frame_id <= self._warmup_frames:
            extra_expand *= 1.30
        if speed > 1.5:
            extra_expand *= min(1.50, 1.0 + 0.07 * speed)
        if self._low_conf_streak > 0:
            extra_expand *= min(1.90, 1.18 + 0.18 * self._low_conf_streak)

        search_rect = self._get_padded_roi(self._roi, scale=1.0, center_override=predicted_center, extra_expand=extra_expand)
        peak_x, peak_y, scale_factor, best_peak, best_psr = self.track(search_rect, image)

        delta = np.array([peak_x, peak_y], dtype=np.float32) - self._tmpl_sz / 2.0
        search_scale = np.array(search_rect[2:], dtype=np.float32) / np.array(self._tmpl_sz, dtype=np.float32)
        delta = delta * search_scale
        new_center = predicted_center + delta

        old_size = np.array([self._roi[2], self._roi[3]], dtype=np.float32)
        confident = (best_peak > self._peak_threshold) and (best_psr > self._psr_threshold)
        very_confident = (best_peak > self._peak_threshold + 0.01) and (best_psr > self._good_psr)

        # 对 track 选出的尺度做平滑，不搞“放大优先”，而是缓慢靠近最优候选
        blended_ratio = 1.0 + (scale_factor - 1.0) * self._scale_smooth
        blended_ratio = float(np.clip(blended_ratio, 1.0 - self._max_shrink_per_frame, 1.0 + self._max_grow_per_frame))

        # 抗收缩：只有很自信且连续几帧都支持缩小，才真正缩
        if blended_ratio < 1.0:
            if very_confident and scale_factor < 0.985:
                self._shrink_streak += 1
            else:
                self._shrink_streak = 0
            if self._shrink_streak < 2:
                blended_ratio = 1.0
        else:
            self._shrink_streak = 0

        new_size = old_size * blended_ratio
        if self._init_size is not None:
            new_size = np.maximum(new_size, self._init_size * self._min_size_ratio)
        if self._size_ref is not None:
            # 防止长期收缩到局部亮点，但只是软保护，不去强推放大
            new_size = np.maximum(new_size, self._size_ref * 0.88)
        new_size = np.maximum(new_size, 1.0)

        self._roi = self._make_roi_from_center(new_center, new_size)
        self._roi[0] = min(max(self._roi[0], 0.0), max(0.0, w_img - self._roi[2]))
        self._roi[1] = min(max(self._roi[1], 0.0), max(0.0, h_img - self._roi[3]))

        measured_center = self._roi_center(self._roi)
        raw_velocity = measured_center - current_center
        if confident:
            self._velocity = 0.60 * self._velocity + 0.40 * raw_velocity
            self._low_conf_streak = 0
            ref_lr = 0.035 if very_confident else 0.018
            self._size_ref = (1.0 - ref_lr) * self._size_ref + ref_lr * np.array([self._roi[2], self._roi[3]], dtype=np.float32)

            x = self.getFeatures(image, self._get_padded_roi(self._roi, scale=1.0, extra_expand=1.03), self._tmpl_sz)
            lr = self._interp_factor if not very_confident else min(0.05, self._interp_factor * 1.2)
            self.update_model(x, lr)
        else:
            self._low_conf_streak += 1
            self._velocity *= 0.82

        return self._roi
