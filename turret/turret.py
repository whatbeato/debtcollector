# playwright install chromium please
# k thx bai
import cv2
import threading
import mediapipe as mp
import serial
import re
import sys
import logging
from datetime import date
from time import sleep, time
from pynput import keyboard
from playwright.sync_api import sync_playwright

# ── Configuration ──────────────────────────────────────────────────────────────
FRAME_W = 1280
FRAME_H = 720
CENTER_X = FRAME_W // 2
CENTER_Y = FRAME_H // 2

X_DEADBAND = 60
Y_DEADBAND = 40
X_MIN, X_MAX = -10.0, 10.0
Y_MIN, Y_MAX = 30.0, 140.0
SMOOTH_ALPHA = 0.25

# Platform-specific — update for your OS:
#   Windows: "COM5"  |  Linux: "/dev/ttyUSB0"  |  macOS: "/dev/cu.usbserial-…"
SERIAL_PORT = "COM5"
CAMERA_INDEX = 1

SPEED_NORMAL = 1000
SPEED_FAST = 2000

# Windows path — update for your OS and profile location
USER_DATA_DIR = r"C:\turret\amazon-profile"
PRODUCT_SEARCH = "rubber chicken"

# PID gains for X (pan)
X_KP = 0.2
X_KI = 0.0005
X_KD = 0.0001
X_INTEGRAL_LIMIT = 30.0

# PID gains for Y (tilt)
Y_KP = 0.005
Y_KI = 0.0005
Y_KD = 0.0002
Y_INTEGRAL_LIMIT = 30.0

_CAMERA_READ_RETRIES = 5

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("turret")

# ── Shared state ───────────────────────────────────────────────────────────────
# _face_lock guards _face_cx, _face_cy, _face_detected which are written by
# FindFace and read by MoveLight on different threads.
_face_lock = threading.Lock()
_face_cx = float(CENTER_X)
_face_cy = float(CENTER_Y)
_face_detected = False

tracking_event = threading.Event()
tracking_event.set()

# Set by main after FindFace exits to signal MoveLight to stop.
_stop_event = threading.Event()


# ── Helpers ────────────────────────────────────────────────────────────────────
def _mask_pan(pan: str) -> str:
    return "*" * (len(pan) - 4) + pan[-4:]


def _luhn_check(pan: str) -> bool:
    if not pan.isdigit():
        return False
    digits = [int(d) for d in reversed(pan)]
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _is_expired(exp_month: str, exp_year: str) -> bool:
    try:
        today = date.today()
        year = 2000 + int(exp_year)
        month = int(exp_month)
        return year < today.year or (year == today.year and month < today.month)
    except ValueError:
        return True


# ── PID controller ─────────────────────────────────────────────────────────────
class PID:
    def __init__(self, kp, ki, kd, integral_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, error, dt):
        self._integral = max(
            -self.integral_limit,
            min(self.integral_limit, self._integral + error * dt),
        )
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0


# ── Face detection thread ──────────────────────────────────────────────────────
def FindFace(cam):
    global _face_cx, _face_cy, _face_detected

    mpFaceDetection = mp.solutions.face_detection
    mpDraw = mp.solutions.drawing_utils
    faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.1)

    smooth_cx = float(CENTER_X)
    smooth_cy = float(CENTER_Y)

    while not _stop_event.is_set():
        ret, frame = cam.read()
        if not ret:
            for _ in range(_CAMERA_READ_RETRIES):
                sleep(0.1)
                ret, frame = cam.read()
                if ret:
                    break
            if not ret:
                log.error("Camera read failed repeatedly, exiting FindFace.")
                break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)

        if results.detections:
            detection = results.detections[0]
            score = round(detection.score[0] * 100)

            mpDraw.draw_detection(frame, detection)
            bboxC = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            fx = int(bboxC.xmin * w)
            fy = int(bboxC.ymin * h)
            bw = int(bboxC.width * w)
            bh = int(bboxC.height * h)
            cx = fx + bw // 2
            cy = fy + bh // 2

            smooth_cx = SMOOTH_ALPHA * cx + (1 - SMOOTH_ALPHA) * smooth_cx
            smooth_cy = SMOOTH_ALPHA * cy + (1 - SMOOTH_ALPHA) * smooth_cy

            with _face_lock:
                _face_cx = smooth_cx
                _face_cy = smooth_cy
                _face_detected = True

            cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            cv2.rectangle(frame, (fx, fy), (fx + bw, fy + bh), (0, 255, 0), 2)
            cv2.putText(frame, f"{score}%", (fx, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            with _face_lock:
                _face_detected = False

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


# ── Motor control thread ───────────────────────────────────────────────────────
def MoveLight(ser):
    pid_x = PID(X_KP, X_KI, X_KD, X_INTEGRAL_LIMIT)
    pid_y = PID(Y_KP, Y_KI, Y_KD, Y_INTEGRAL_LIMIT)
    last_time = time()
    prev_in_deadband_x = True
    prev_in_deadband_y = True
    x = 0.0
    y = 85.0
    motor_speed = SPEED_NORMAL

    while not _stop_event.is_set():
        with _face_lock:
            detected = _face_detected
            cx = _face_cx
            cy = _face_cy

        if not tracking_event.is_set() or not detected:
            pid_x.reset()
            pid_y.reset()
            prev_in_deadband_x = True
            prev_in_deadband_y = True
            last_time = time()
            sleep(0.05)
            continue

        now = time()
        dt = max(now - last_time, 1e-6)
        last_time = now

        error_x = cx - CENTER_X
        error_y = cy - CENTER_Y

        in_deadband_x = abs(error_x) <= X_DEADBAND
        if not in_deadband_x:
            x = max(X_MIN, min(X_MAX, x + pid_x.compute(error_x, dt)))
        elif not prev_in_deadband_x:
            pid_x.reset()
        prev_in_deadband_x = in_deadband_x

        in_deadband_y = abs(error_y) <= Y_DEADBAND
        if not in_deadband_y:
            # face below center (error_y > 0) → tilt down → y decreases
            y = max(Y_MIN, min(Y_MAX, y - pid_y.compute(error_y, dt)))
        elif not prev_in_deadband_y:
            pid_y.reset()
        prev_in_deadband_y = in_deadband_y

        if in_deadband_x and in_deadband_y:
            motor_speed = SPEED_FAST
        else:
            motor_speed = SPEED_NORMAL

        command = "{} {} {}\n".format(round(x), round(y), motor_speed)
        try:
            ser.write(command.encode("utf-8"))
        except serial.SerialException as e:
            log.error("Serial write failed: %s — stopping MoveLight.", e)
            break
        log.debug("x=%.2f y=%.2f spd=%d err=(%.0f,%.0f)", x, y, motor_speed, error_x, error_y)
        sleep(0.05)


# ── Card reader ────────────────────────────────────────────────────────────────
class CardReader:
    _TRACK1_RE = re.compile(r"%B(\d{13,19})\^([^\^]+)\^(\d{2})(\d{2})")

    def __init__(self, on_swipe, tracking_event):
        self._on_swipe = on_swipe
        self._tracking_event = tracking_event
        self._buf = None
        self._fired = False
        self._listener = keyboard.Listener(on_press=self._on_press)

    def _on_press(self, key):
        if self._fired:
            return
        try:
            ch = key.char
        except AttributeError:
            if key == keyboard.Key.enter and self._buf is not None:
                self._process()
            return
        if ch is None:
            return
        if ch == '%':
            self._buf = '%'
            self._tracking_event.clear()
            return
        if self._buf is not None:
            self._buf += ch
            if self._buf.count('?') >= 2:
                self._process()

    def _process(self):
        raw = self._buf
        self._buf = None
        m = self._TRACK1_RE.search(raw)
        if not m:
            log.warning("[CardReader] Malformed swipe: could not parse track data.")
            self._tracking_event.set()
            return
        pan = m.group(1)
        name = m.group(2).replace('/', ' ').strip()
        exp_year = m.group(3)
        exp_month = m.group(4)

        if not _luhn_check(pan):
            log.warning("[CardReader] Card %s failed Luhn check — ignoring.", _mask_pan(pan))
            self._tracking_event.set()
            return

        if _is_expired(exp_month, exp_year):
            log.warning("[CardReader] Card %s is expired (%s/%s) — ignoring.",
                        _mask_pan(pan), exp_month, exp_year)
            self._tracking_event.set()
            return

        log.info("[CardReader] Valid card read: %s, holder: %s", _mask_pan(pan), name)
        self._fired = True
        self._on_swipe({"pan": pan, "name": name, "exp_month": exp_month, "exp_year": exp_year})

    def reset(self):
        self._buf = None
        self._fired = False
        self._tracking_event.set()

    def start(self):
        self._listener.start()


# ── Amazon purchase ────────────────────────────────────────────────────────────
def run_amazon_purchase(card):
    try:
        with sync_playwright() as p:
            ctx = p.chromium.launch_persistent_context(USER_DATA_DIR, headless=False)
            try:
                page = ctx.new_page()

                # Search and add to cart
                page.goto("https://www.amazon.com/POPLAY-Rubber-Chicken-Squeeze-Novelty/dp/B01LYW69OL/")
                page.wait_for_load_state("domcontentloaded")
                page.wait_for_selector("#add-to-cart-button")
                page.locator("#add-to-cart-button").click()
                page.wait_for_selector("#NATC_SMART_WAGON_CONF_MSG_SUCCESS, #sw-gtc", timeout=10000)

                # Navigate to cart and proceed to checkout
                page.goto("https://www.amazon.com/gp/cart/view.html")
                page.wait_for_load_state("domcontentloaded")
                page.get_by_role("button", name=re.compile("Proceed to checkout", re.I)).click()
                page.wait_for_load_state("domcontentloaded")
                # "Continue to checkout" only appears when Amazon shows an intermediate sign-in page
                continue_link = page.get_by_role("link", name="Continue to checkout")
                if continue_link.count() > 0:
                    continue_link.first.click()
                    page.wait_for_load_state("domcontentloaded")

                # Select 15 Falls Rd address, HCB finna enjoy this one :bangbang:
                page.get_by_role("button", name="Show more addresses").click()
                page.get_by_text("15 FALLS RD, SHELBURNE, VT,").click()
                page.get_by_test_id("bottom-continue-button").click()

                # Payment — add the swiped card
                page.get_by_role("link", name="Change payment method").click()
                page.wait_for_load_state("domcontentloaded")
                page.get_by_role("link", name="Add a credit or debit card").click()
                page.wait_for_selector("iframe[name^='ApxSecureIframe']", timeout=15000)
                iframe = page.frame_locator("iframe[name^='ApxSecureIframe']")
                iframe.get_by_role("textbox", name="Card number").fill(card["pan"])
                iframe.get_by_role("textbox", name="Name on card").fill(card["name"])
                # The payment iframe uses native <select> elements for month/year even when
                # Amazon skins them with a custom dropdown UI — select_option reaches through.
                iframe.locator("select").nth(0).select_option(card["exp_month"])
                iframe.locator("select").nth(1).select_option("20" + card["exp_year"])
                iframe.get_by_role("button", name="Add your card").click()
                sleep(2)
                page.locator("iframe[name=\"ApxSecureIframe-pp-SpyOEA-8\"]").content_frame.locator("input[name=\"ppw-widgetEvent:SavePaymentMethodDetailsEvent\"]").click()
                page.wait_for_load_state("domcontentloaded")

                # Select the newly added card by its last 4 digits, then place order
                page.get_by_text(f"ending in {card['pan'][-4:]}", exact=True).click()
                page.get_by_test_id("bottom-continue-button").click()
                page.wait_for_load_state("domcontentloaded")
                page.locator("#bottomSubmitOrderButtonId").get_by_test_id("SPC_selectPlaceOrder").click()
                sleep(7) # wifi is too slow at this venue ts sucks lol
                log.info("[Amazon] Order placed successfully for %s", card["name"])

            finally:
                ctx.close()
    except Exception as e:
        log.error("[Amazon] Playwright error: %s", e)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            log.error("Failed to open camera at index %d.", CAMERA_INDEX)
            sys.exit(1)
    except Exception as e:
        log.error("Camera initialization failed: %s", e)
        sys.exit(1)

    try:
        ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
    except serial.SerialException as e:
        log.error("Failed to open serial port %s: %s", SERIAL_PORT, e)
        cap.release()
        sys.exit(1)

    def handle_swipe(card):
        def _run():
            run_amazon_purchase(card)
            reader.reset()
        threading.Thread(target=_run, daemon=True).start()

    reader = CardReader(on_swipe=handle_swipe, tracking_event=tracking_event)
    reader.start()

    t1 = threading.Thread(target=FindFace, args=(cap,), name="FindFace")
    t2 = threading.Thread(target=MoveLight, args=(ser,), name="MoveLight", daemon=True)
    t1.start()
    t2.start()
    try:
        t1.join()
    finally:
        _stop_event.set()
        t2.join(timeout=2.0)
        ser.close()




