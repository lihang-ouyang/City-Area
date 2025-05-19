import cv2
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 圖片載入
original_img = cv2.imread("map.jpg")
gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

# 圖片二值化（反轉，讓區塊內是黑、邊線是白）
_, thresh = cv2.threshold(255 - gray, 127, 255, cv2.THRESH_BINARY)

# 找所有輪廓（區塊邊界）
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 儲存主圖與顯示圖
display_img = original_img.copy()
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

            selected = cv2.bitwise_and(original_img, original_img, mask=mask)
            cv2.imshow("Selected Area", selected)
            draw_integration_figures(mask)

    elif event == cv2.EVENT_RBUTTONDOWN:
        hover_index = -1
        display_img = original_img.copy()

def draw_integration_figures(mask, scale_km_per_pixel=None):
    import matplotlib.pyplot as plt

    # 擷取區域
    ys, xs = np.where(mask > 0)
    y0, y1 = np.min(ys), np.max(ys)
    x0, x1 = np.min(xs), np.max(xs)
    region_mask = mask[y0:y1+1, x0:x1+1]
    region_img = original_img[y0:y1+1, x0:x1+1].copy()
    h, w = region_mask.shape

    bar_width = max(8, w // 40)   # 長條寬度
    color_upper = (0, 0, 255)     # Blue for upper
    color_central = (255, 0, 255) # Magenta for central
    color_lower = (255, 128, 0)   # Orange for lower

    # 1. 上矩形法 (每bar最高點)
    upper_img = region_img.copy()
    area_upper = 0
    for i in range(0, w, bar_width):
        # 在這個bar範圍內收集所有有效像素點
        bar_cols = region_mask[:, i:i+bar_width]
        ys_bar, xs_bar = np.where(bar_cols > 0)
        if len(ys_bar):
            top = ys_bar.min()
            bot = ys_bar.max()
            max_height = bot - top + 1
            # 取這個bar裡最高點作為高度
            rect_top = top
            rect_bot = top + max_height - 1
            cv2.rectangle(
                upper_img,
                (i, rect_top),
                (min(i + bar_width - 1, w - 1), rect_bot),
                color_upper, 1
            )
            area_upper += max_height * min(bar_width, w-i)
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(upper_img, contours, -1, (0,255,0), 1)

    # 2. 中矩形法 (最高最低的平均)
    central_img = region_img.copy()
    area_central = 0
    for i in range(0, w, bar_width):
        bar_cols = region_mask[:, i:i+bar_width]
        ys_bar, xs_bar = np.where(bar_cols > 0)
        if len(ys_bar):
            top = ys_bar.min()
            bot = ys_bar.max()
            avg = int((top + bot) / 2)
            height = bot - top + 1
            rect_top = int(avg - height/2)
            rect_bot = int(avg + (height-1)/2)
            cv2.rectangle(
                central_img,
                (i, rect_top),
                (min(i + bar_width - 1, w - 1), rect_bot),
                color_central, 1
            )
            area_central += height * min(bar_width, w-i)
    cv2.drawContours(central_img, contours, -1, (0,255,0), 1)

    # 3. 下矩形法 (每bar最低點)
    lower_img = region_img.copy()
    area_lower = 0
    for i in range(0, w, bar_width):
        bar_cols = region_mask[:, i:i+bar_width]
        ys_bar, xs_bar = np.where(bar_cols > 0)
        if len(ys_bar):
            top = ys_bar.min()
            bot = ys_bar.max()
            min_height = bot - top + 1
            rect_bot = bot
            rect_top = bot - min_height + 1
            cv2.rectangle(
                lower_img,
                (i, rect_top),
                (min(i + bar_width - 1, w - 1), rect_bot),
                color_lower, 1
            )
            area_lower += min_height * min(bar_width, w-i)
    cv2.drawContours(lower_img, contours, -1, (0,255,0), 1)

    # 4. 梯形法（同前，不動此段）
    trap_img = region_img.copy()
    area_trap = 0
    color_trap = (255, 0, 0)  # Blue
    for i in range(0, w-bar_width, bar_width):
        bar_cols1 = region_mask[:, i:i+bar_width]
        bar_cols2 = region_mask[:, i+bar_width:i+2*bar_width]
        ys_bar1, xs_bar1 = np.where(bar_cols1 > 0)
        ys_bar2, xs_bar2 = np.where(bar_cols2 > 0)
        if len(ys_bar1) and len(ys_bar2):
            top1 = ys_bar1.min()
            bot1 = ys_bar1.max()
            top2 = ys_bar2.min()
            bot2 = ys_bar2.max()
            l, r = i, i+bar_width
            # 上底
            cv2.line(trap_img, (l, top1), (r, top2), color_trap, 1)
            # 下底
            cv2.line(trap_img, (l, bot1), (r, bot2), color_trap, 1)
            # 左右
            cv2.line(trap_img, (l, top1), (l, bot1), color_trap, 1)
            cv2.line(trap_img, (r, top2), (r, bot2), color_trap, 1)
            avg_height = ((bot1 - top1 + 1) + (bot2 - top2 + 1)) / 2
            area_trap += avg_height * bar_width
    cv2.drawContours(trap_img, contours, -1, (0,255,0), 1)

    # 5. 蒙特卡洛法
    monte_img = region_img.copy()
    N = 3000
    in_cnt = 0
    rng = np.random.default_rng()
    for _ in range(N):
        x = rng.integers(0, w)
        y = rng.integers(0, h)
        if region_mask[y, x] > 0:
            monte_img[y, x] = [255,0,0]
            in_cnt += 1
        else:
            monte_img[y, x] = [0,128,255]
    area_mc = in_cnt / N * (w * h)
    cv2.drawContours(monte_img, contours, -1, (0,255,0), 1)

    area_true = np.sum(region_mask > 0)
    # 顯示
    fig, axs = plt.subplots(1, 5, figsize=(22,4))
    axs[0].imshow(cv2.cvtColor(upper_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title(f"上矩形法\n最高點 Darboux\n積分={area_upper} px²")
    axs[1].imshow(cv2.cvtColor(central_img, cv2.COLOR_BGR2RGB))
    axs[1].set_title(f"中矩形法\n最高+最低平均\n積分={area_central} px²")
    axs[2].imshow(cv2.cvtColor(lower_img, cv2.COLOR_BGR2RGB))
    axs[2].set_title(f"下矩形法\n最低點 Darboux\n積分={area_lower} px²")
    axs[3].imshow(cv2.cvtColor(trap_img, cv2.COLOR_BGR2RGB))
    axs[3].set_title(f"梯形法\n積分={area_trap:.0f} px²")
    axs[4].imshow(cv2.cvtColor(monte_img, cv2.COLOR_BGR2RGB))
    axs[4].set_title(f"蒙特卡洛法\n積分={area_mc:.0f} px²\n(真實={area_true} px²)")
    for ax in axs: ax.axis("off")
    plt.tight_layout()
    plt.show()

# 設定視窗與滑鼠事件
cv2.namedWindow("Map")
cv2.setMouseCallback("Map", on_mouse)

while True:
    if cv2.getWindowProperty("Map", cv2.WND_PROP_VISIBLE) < 1:
        break

    temp = original_img.copy()
    if hover_index != -1:
        cv2.drawContours(temp, [contours[hover_index]], -1, (0, 0, 255), 2)

    cv2.imshow("Map", temp)
    key = cv2.waitKey(20)
    if key == 27:
        break
    elif key == ord('r'):
        hover_index = -1

cv2.destroyAllWindows()
