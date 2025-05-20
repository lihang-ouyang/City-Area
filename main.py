import cv2
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 設定最大顯示視窗大小
MAX_WINDOW_WIDTH = 800
MAX_WINDOW_HEIGHT = 600

# 圖片載入
original_img = cv2.imread("map2.jpg")
original_height, original_width = original_img.shape[:2]

# 計算縮放比例
scale_width = MAX_WINDOW_WIDTH / original_width
scale_height = MAX_WINDOW_HEIGHT / original_height
scale = min(scale_width, scale_height, 1)  # 確保縮放比例不超過1（即不放大圖片）

# 調整圖片大小
resized_img = cv2.resize(original_img, (int(original_width * scale), int(original_height * scale)))
gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

# 圖片二值化（反轉，讓區塊內是黑、邊線是白）
_, thresh = cv2.threshold(255 - gray, 127, 255, cv2.THRESH_BINARY)

# 找所有輪廓（區塊邊界）
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 計算整個台灣的南北長像素距離
ys, _ = np.where(thresh > 0)
y_min, y_max = ys.min(), ys.max()
taiwan_pixel_distance = y_max - y_min + 1  # 台灣南北長度的總像素數
taiwan_length_km = 394  # 台灣南北實際長度（公里）
scale_km_per_pixel = taiwan_length_km / taiwan_pixel_distance  # 固定比例尺
print(f"台灣南北長（像素）: {taiwan_pixel_distance} px")
print(f"固定比例尺: {scale_km_per_pixel:.6f} 公里/像素")

# 儲存主圖與顯示圖
display_img = resized_img.copy()
hover_index = -1  # 當前滑鼠指到哪個輪廓

def on_mouse(event, x, y, flags, param):
    global hover_index, display_img

    if event == cv2.EVENT_MOUSEMOVE:
        found = False
        for i, cnt in enumerate(contours):
            if cv2.pointPolygonTest(cnt, (x, y), False) >= 0:
                hover_index = i
                found = True
                break
        if not found:
            hover_index = -1

    elif event == cv2.EVENT_LBUTTONDOWN:
        if hover_index != -1:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contours[hover_index]], -1, 255, thickness=-1)

            selected = cv2.bitwise_and(resized_img, resized_img, mask=mask)
            cv2.imshow("Selected Area", selected)

            # 繪製積分圖
            draw_integration_figures(mask)

    elif event == cv2.EVENT_RBUTTONDOWN:
        hover_index = -1
        display_img = resized_img.copy()

def draw_integration_figures(mask):
    import matplotlib.pyplot as plt

    # 擷取區域
    ys, xs = np.where(mask > 0)
    y0, y1 = np.min(ys), np.max(ys)
    x0, x1 = np.min(xs), np.max(xs)
    region_mask = mask[y0:y1+1, x0:x1+1]
    region_img = resized_img[y0:y1+1, x0:x1+1].copy()
    h, w = region_mask.shape

    # 蒙特卡洛法
    monte_img = region_img.copy()
    area = w * h  # 總範圍面積
    region_area_px2 = np.sum(region_mask > 0)  # 真實區域面積（像素）

    # 換算平方公里
    region_area_km2 = region_area_px2 * (scale_km_per_pixel ** 2)

    # 根據區域大小動態調整點數量
    base_density = 50  # 每單位區域的基礎點數密度
    scaling_factor = 0.5  # 小區域點數減少的比例
    N = max(1000, int(base_density * (region_area_px2 ** scaling_factor)))  # 點數動態調整
    in_cnt = 0
    rng = np.random.default_rng()

    for _ in range(N):
        x = rng.integers(0, w)
        y = rng.integers(0, h)
        if region_mask[y, x] > 0:
            monte_img[y, x] = [255, 0, 0]  # 紅色點（區域內）
            in_cnt += 1
        else:
            monte_img[y, x] = [0, 128, 255]  # 藍色點（區域外）

    monte_area_km2 = in_cnt / N * area * (scale_km_per_pixel ** 2)
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(monte_img, contours, -1, (0, 255, 0), 1)  # 綠色邊界

    # 顯示（僅蒙特卡洛法，調整圖像大小和標題）
    fig, ax = plt.subplots(figsize=(7, 7))  # 縮小圖像大小
    ax.imshow(cv2.cvtColor(monte_img, cv2.COLOR_BGR2RGB))
    ax.set_title(
        f"蒙特卡洛法積分={monte_area_km2:.2f} km²\n真實面積={region_area_km2:.2f} km²\n固定比例尺: {scale_km_per_pixel:.6f} 公里/像素",
        fontsize=12
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()

# 設定視窗與滑鼠事件
cv2.namedWindow("Map")
cv2.setMouseCallback("Map", on_mouse)

while True:
    if cv2.getWindowProperty("Map", cv2.WND_PROP_VISIBLE) < 1:
        break

    temp = resized_img.copy()
    if hover_index != -1:
        cv2.drawContours(temp, [contours[hover_index]], -1, (0, 0, 255), 2)

    cv2.imshow("Map", temp)
    key = cv2.waitKey(20)
    if key == 27:
        break
    elif key == ord('r'):
        hover_index = -1

cv2.destroyAllWindows()