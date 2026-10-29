import pyautogui, cv2, numpy as np, time, os, keyboard
from ultralytics import YOLO

# ---------- аварийный выход ----------
pyautogui.FAILSAFE = True 

def emergency_exit():
    print('\nАварийная остановка (пробел)')
    os._exit(0)

keyboard.add_hotkey('space', emergency_exit)
# --------------------------------------

MODEL_PATH = 'runs/detect/cookie_detector-4/weights/best.pt'
model = YOLO(MODEL_PATH)
print(model.names)

DETECT_CONF = 0.25
CARD_MIN = 0.4 
AD_MIN, GAME_MIN = 0.5, 0.6
ACTIVE_SCORE = 235  

last = {'img': None, 'cards': []}

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)

def detect():
    shot = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    res = model(img, conf=DETECT_CONF, verbose=False)[0]
    objs = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        objs.append({'name': model.names[int(b.cls[0])],
                     'box': (x1, y1, x2, y2),
                     'conf': float(b.conf[0])})
    return img, objs

def center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

def get_cards(img, objs):
    """Все карточки магазина (любой из двух классов) + яркость каждой."""
    cand = [o for o in objs
            if o['name'] in ('upgrade', 'upgrade_locked') and o['conf'] >= CARD_MIN]
    cand.sort(key=lambda o: -o['conf'])
    cards = []
    for o in cand:
        if any(iou(o['box'], c['box']) > 0.5 for c in cards):
            continue 
        x1, y1, x2, y2 = [max(int(v), 0) for v in o['box']]
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        score = float(np.percentile(gray, 95))
        cards.append({**o, 'score': score, 'active': score >= ACTIVE_SCORE})
    return sorted(cards, key=lambda c: c['box'][1])

def debug_dump():
    if last['img'] is None:
        return
    img = last['img'].copy()
    print('\n--- карточки ---')
    for c in last['cards']:
        x1, y1, x2, y2 = [int(v) for v in c['box']]
        color = (0, 255, 0) if c['active'] else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        cv2.putText(img, f"{c['name']} {c['conf']:.2f} s={c['score']:.0f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        print(f"{c['name']:15} conf={c['conf']:.2f} score={c['score']:.0f} active={c['active']}")
    cv2.imwrite('debug.png', img)
    print('Сохранено: debug.png (зелёные - будет покупать, красные - пропустит)')

keyboard.add_hotkey('f9', debug_dump)  

def main():
    print('ПРОБЕЛ - аварийный выход, F9 - отладка (debug.png и таблица карточек)')
    while True:
        img, objs = detect()
        cards = get_cards(img, objs)
        last['img'], last['cards'] = img, cards

        # 1. реклама
        ads = [o for o in objs if o['name'] == 'close_ad' and o['conf'] >= AD_MIN]
        if ads:
            pyautogui.click(*center(max(ads, key=lambda o: o['conf'])['box']))
            time.sleep(0.5)
            continue

        
        active = [c for c in cards if c['active']]
        if active:
            target = active[-1]
            pyautogui.click(*center(target['box']))
            print('Покупка:', [int(v) for v in center(target['box'])],
                  f"score={target['score']:.0f}")
            time.sleep(0.4)
            continue

        # 3. доллар
        games = [o for o in objs if o['name'] == 'game' and o['conf'] >= GAME_MIN]
        if games:
            x, y = center(max(games, key=lambda o: o['conf'])['box'])
            for _ in range(10):
                pyautogui.click(x, y)
                time.sleep(0.05)
        time.sleep(0.2)

try:
    main()
except pyautogui.FailSafeException:
    print('\nАварийная остановка (мышь в углу экрана)')
except KeyboardInterrupt:
    print('\nОстановлено (Ctrl+C)')