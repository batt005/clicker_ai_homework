import pyautogui, cv2, numpy as np, time, os, keyboard
from ultralytics import YOLO


UPGRADE_COOLDOWN = 2.5
CLICK_BURST = 1.0
CLICK_INTERVAL = 0.01

DETECT_CONF = 0.25
CARD_MIN = 0.3
AD_MIN, GAME_MIN = 0.5, 0.6
WHITE_LEVEL = 235
WHITE_RATIO_MIN = 0.01
MODEL_PATH = 'runs/detect/cookie_detector-4/weights/best.pt'


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

def emergency_exit():
    print('\nАварийная остановка (пробел)')
    os._exit(0)

keyboard.add_hotkey('space', emergency_exit)


model = YOLO(MODEL_PATH)
print(model.names)

last = {'img': None, 'cards': [], 'objs': []}
last_buy_time = 0.0

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
    """Все карточки магазина (любой из двух классов) + доля белых пикселей в каждой."""
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

        white = float((crop.min(axis=2) >= WHITE_LEVEL).mean())
        cards.append({**o, 'white': white, 'active': white >= WHITE_RATIO_MIN})
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
        cv2.putText(img, f"{c['name']} {c['conf']:.2f} w={c['white']*100:.1f}%",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        print(f"{c['name']:15} conf={c['conf']:.2f} белых={c['white']*100:.1f}% active={c['active']}")
    low = [o for o in last['objs']
           if o['name'] in ('upgrade', 'upgrade_locked') and o['conf'] < CARD_MIN]
    if low:
        print(f'Ещё найдено слабых карточек (conf < {CARD_MIN}): {len(low)}',
              [f"{o['name']} {o['conf']:.2f}" for o in low])
    cv2.imwrite('debug.png', img)
    print('Сохранено: debug.png (зелёные - будет покупать, красные - пропустит)')

keyboard.add_hotkey('f9', debug_dump)

def click_burst(x, y, seconds):
    """Быстрая серия кликов: мышь наводим один раз, дальше только клики."""
    pyautogui.moveTo(x, y)
    end = time.time() + seconds
    n = 0
    while time.time() < end:
        pyautogui.click()
        n += 1
        if CLICK_INTERVAL:
            time.sleep(CLICK_INTERVAL)
    return n

def main():
    global last_buy_time
    print('ПРОБЕЛ - аварийный выход, F9 - отладка (debug.png и таблица карточек)')
    while True:
        img, objs = detect()
        cards = get_cards(img, objs)
        last['img'], last['cards'], last['objs'] = img, cards, objs

    
        ads = [o for o in objs if o['name'] == 'close_ad' and o['conf'] >= AD_MIN]
        if ads:
            pyautogui.click(*center(max(ads, key=lambda o: o['conf'])['box']))
            time.sleep(0.5)
            continue

       
        active = [c for c in cards if c['active']]
        if active and time.time() - last_buy_time >= UPGRADE_COOLDOWN:
            target = active[-1]
            pyautogui.click(*center(target['box']))
            last_buy_time = time.time()
            print('Покупка:', [int(v) for v in center(target['box'])],
                  f"белых={target['white']*100:.1f}%")


        games = [o for o in objs if o['name'] == 'game' and o['conf'] >= GAME_MIN]
        if games:
            x, y = center(max(games, key=lambda o: o['conf'])['box'])
            n = click_burst(x, y, CLICK_BURST)
            print(f'Кликов по доллару: {n}')
        else:
            time.sleep(0.2)

try:
    main()
except pyautogui.FailSafeException:
    print('\nАварийная остановка (мышь в углу экрана)')
except KeyboardInterrupt:
    print('\nОстановлено (Ctrl+C)')
