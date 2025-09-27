# play_and_detect.py — continuous playback with fall/walk HUD using lstm_model
# Controls: Space = pause/resume, q/ESC = quit
# Logging: use --log-csv to dump per-frame metrics for later analysis

# ** NOTE **
# How to run?
# -> python play_and_detect.py --video test4.mp4 --use-model

import os, sys, cv2, glob, argparse, time, csv, math
# os: lets you talk to the computer’s operating system. sys: lets you talk to Python itself
# cv2: this is OpenCV, a big library for working with images and videos. we use it to:
        # open the video file, draw the green dots etc., show the video window on screen.
# glob: this is like a “file finder.” If you tell it "*.mp4", it finds all .mp4 video files in a folder.
# argparse: lets you add command-line options, like --video myclip.mp4 or --log-csv
# csv: lets you write your results into a .csv file

import numpy as np
from collections import deque
# deque is a special list you can add to on one end and remove from the other, very quickly.
# We use a deque to keep a sliding window of the last few frames (like the last 45 skeletons). That way we always analyze a moving time window without storing the entire video.

from datetime import datetime

# =========================
# =========================

FRAMES_PER_WINDOW = 45     # How many frames we look at together as one 'window' when deciding walk/fall. Temporal context per prediction (lower = faster reaction)
HUD = True                 # draw status banner
DRAW_POINTS = True         # draw pose keypoints (green dots)
SMOOTH_WINDOW = 7          # moving average length for smoothed prob. We average the last this-many('6') probabilities to keep the label from flickering.

# Debounce / dwell (helps avoid early triggers)
ENTRY_DWELL = 5            # We only say 'FALL' if the trigger stays true for this many frames in a row (prevents jumpy false alarms).

# Hysteresis thresholds (conservative default; tune per dataset)
FALL_PROB_ON  = 0.15       # To switch into FALL, the average 'fall' probability must be at least this number
FALL_PROB_OFF = 0.10       # To switch back to WALK, the average 'fall' probability must drop below this number

# Heuristic fusion (helps at impact)
USE_HEURISTIC_FUSION = True # If True, we blend the model's prediction with a hand-made rule (heuristic).
HEURISTIC_WEIGHT = 0.34    # How much the heuristic counts in the blend. 0.34 means 34% heuristic + 66% model.
HEURISTIC_FORCE_ON = 0.46  # If the heuristic is very sure (above this number), allow an instant trigger (still respects dwell).

# Optional conservative dual-gate: BOTH model & heuristic must be decent
DUAL_GATE  = True # If True, be extra careful: require both model and heuristic to be 'good enough' at the same time.
MODEL_GATE = 0.30 # The model's minimum score needed (when DUAL_GATE = True).
HEUR_GATE  = 0.38 # The heuristic's minimum score needed (when HEUR_GATE = True).

# Fallback / defaults if model doesn't provide names
DEFAULT_CLASS_NAMES = ["walk", "fall", "stand", "jump"]

# For HUD color mapping
POINT_RADIUS = 3
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX # Picked an OpenCV font to draw text.

# =========================
# Utility
# =========================
def has_gui(): # Checks if we can open a window on this computer. On some servers there is no screen; this avoids crashing.
    try:
        cv2.namedWindow("__x__"); cv2.destroyWindow("__x__")
        # Asks OpenCV to make a window on your screen with a certain name so we can show the video.
        return True
    except Exception:
        return False

def iter_video_paths(arg): # Figures out which video files to open. It accepts a folder, a single filename, or a pattern like clips/*.mp4.
    if arg is None: # If the user didn’t pass a --video argument…
        if os.path.exists("test3.mp4"):
            yield os.path.abspath("test3.mp4") # Create a full path to a default file named test3.mp4.
        else:
            print("[error] No --video provided and test3.mp4 not found.")
        return
    p = os.path.abspath(arg) # Make the user’s path absolute (full path).
    if os.path.isdir(p): # If p is a folder…
        any_yield = False # Track whether we found any files inside.
        for ext in ("*.mp4","*.mov","*.mkv","*.avi"): # Loop over common video extensions.
            for fp in sorted(glob.glob(os.path.join(p, ext))):
                # os.path.join(p, ext) builds something like "/path/*.mp4".
                # glob.glob(...) lists all files matching that pattern.
                # sorted(...) puts them in alphabetical order.
                yield fp; any_yield = True # Give the file path to the caller. And track that we found at least one.
        if not any_yield: # If the folder had nothing we can use… tell the user.
            print(f"[error] No videos found in folder: {p}")
    else: # If p is not a folder, treat it as a file or wildcard pattern.
        matches = sorted(glob.glob(p)) # Find all matching paths and sort them.
        if not matches:
            print(f"[error] No file(s) match: {arg}")
        for fp in matches:
            if os.path.isfile(fp): yield os.path.abspath(fp)
            # Only keep real files (not folders). Give that file path to the caller.

# =========================
# Pose extraction (MediaPipe) → 25x3 skeleton
# =========================
try: # Start trying to load MediaPipe (pose detection). This may fail if not installed.
    import mediapipe as mp # Load MediaPipe and call it mp. It can detect 33 body landmarks in an image.
    mp_pose = mp.solutions.pose # Grab the “pose” module from MediaPipe.
    _POSE = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                         enable_segmentation=False,
                         min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # Create a pose detector object.
    # static_image_mode=False: optimized for video (tracks across frames).
    # model_complexity=1: medium model.
    # enable_segmentation=False: we don’t need a human mask.

except Exception: # if any of that import fails
    mp = None # Mark MediaPipe as unavailable.
    _POSE = None # Mark the pose object as unavailable.

def _coord(lm): return [lm.x, lm.y, lm.z] # A tiny helper to convert a MediaPipe landmark lm into a simple list [x, y, z].

def mediapipe_to_ntu25(landmarks): # Define a function to convert 33 MediaPipe points into your 25-joint format (the order your model expects).
    if landmarks is None or len(landmarks) < 33: # If pose is missing or incomplete…
        return np.zeros((25,3), dtype=np.float32) # Return a 25×3 table of zeros (meaning “no pose”).
    hip = [(landmarks[23].x + landmarks[24].x) / 2, # Compute mid-hip x by averaging left (23) and right (24) hip x.
           (landmarks[23].y + landmarks[24].y) / 2, # Mid-hip y (average).
           (landmarks[23].z + landmarks[24].z) / 2] # Mid-hip z (average). Now hip is [x,y,z].
    sh  = [(landmarks[11].x + landmarks[12].x) / 2, # Compute mid-shoulder x by averaging left (11) and right (12) shoulders.
           (landmarks[11].y + landmarks[12].y) / 2, # Mid-shoulder y.
           (landmarks[11].z + landmarks[12].z) / 2] # Mid-shoulder z.
    arr = [ # Start a list that will hold exactly 25 [x,y,z] joints in our chosen order.
        hip, [(hip[i]+sh[i])/2 for i in range(3)], sh, _coord(landmarks[0]), # First 4 joints: mid-hip; a midpoint between hip & shoulder (spine-ish); mid-shoulder; nose (landmark 0).
        _coord(landmarks[11]), _coord(landmarks[13]), _coord(landmarks[15]), _coord(landmarks[19]),
        _coord(landmarks[12]), _coord(landmarks[14]), _coord(landmarks[16]), _coord(landmarks[20]),
        _coord(landmarks[23]), _coord(landmarks[25]), _coord(landmarks[27]), _coord(landmarks[31]),
        _coord(landmarks[24]), _coord(landmarks[26]), _coord(landmarks[28]), _coord(landmarks[32]),
        sh, _coord(landmarks[17]), _coord(landmarks[21]), _coord(landmarks[18]), _coord(landmarks[22]),
    ]
    return np.array(arr, dtype=np.float32) # Convert that Python list into a fast NumPy array shaped (25, 3).

def extract_skeleton(frame_bgr): # Define a function: get one frame in BGR, return the 25×3 skeleton (or zeros).
    if _POSE is None: # If we don’t have MediaPipe…
        return np.zeros((25,3), dtype=np.float32) # Return all zeros (no pose).
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) # Convert from OpenCV’s BGR colors to RGB, which MediaPipe expects.
    res = _POSE.process(rgb) # Run the pose detector.
    if res.pose_landmarks: # If it found landmarks…
        return mediapipe_to_ntu25(res.pose_landmarks.landmark) # Convert the 33 landmarks to your 25-joint format and return.
    return np.zeros((25,3), dtype=np.float32) # Otherwise, return zeros.

def draw_points(img, skel25): # Define a function to draw green dots on each joint.
    if not DRAW_POINTS: return img # If drawing is off, just return the image unchanged.
    h, w = img.shape[:2] # Get image height and width (to convert normalized coords to pixel coords).
    for (x,y,z) in skel25: # Loop over each joint’s coordinates.
        if x==0 and y==0 and z==0: continue # Skip joints that are all zeros (no data).
        cv2.circle(img, (int(x*w), int(y*h)), POINT_RADIUS, (0,255,0), -1) # Draw a filled green circle at that joint’s pixel location.
    return img # Give back the image with the drawn dots.

# ---------- NEW: per-frame skeleton normalization (center + scale) ----------
def normalize_skeleton(skel25: np.ndarray) -> np.ndarray: # Define a function: input is a 25×3 skeleton; output is the normalized 25×3 skeleton.
    """
    Center at mid-hip (joint 0) and scale by torso length ||joint2 - joint0||.
    Works per frame; returns a 25x3 array. If skeleton is empty, returns zeros.
    """
    if skel25 is None or skel25.shape != (25,3): # If the input isn’t the right shape, treat it as invalid.
        return np.zeros((25,3), dtype=np.float32) # Return zeros for invalid input.
    if not np.any(skel25): # If all numbers are zero…
        return np.zeros((25,3), dtype=np.float32) # Return zeros.

    center = skel25[0]                # mid-hip per our mapping. Pick the hip joint (index 0) as the center.
    torso_vec = skel25[2] - skel25[0] # mid-shoulder - mid-hip. Compute the torso vector (shoulder minus hip).
    torso_len = float(np.linalg.norm(torso_vec)) + 1e-6 # Get torso length (distance). Add a tiny number so we never divide by zero.

    normed = (skel25 - center) / torso_len # Subtract the center (recenters at 0,0,0) and divide by torso length (standardizes size).
    return normed.astype(np.float32) # Return the normalized skeleton as 32-bit floats.
# ---------------------------------------------------------------------------

# =========================
# LSTM model integration (uses our lstm_model)
# =========================
def try_load_predictor(weights_path=None, debug=False): # Define a function: try to load our model module and its weights, and return the callable functions we need.
    """
    Returns: (predict_fn, preprocess_fn, class_names) or (None, None, None)
    """
    try:
        import importlib
        mod = importlib.import_module("lstm_model") # Import our module named lstm_model
        # Ensure model is loaded (weights)
        if hasattr(mod, "load_model"): # If our module provides a load_model function…
            _ = mod.load_model(weights_path or getattr(mod, "DEFAULT_WEIGHTS", "action_lstm.pth")) # Call it, using either the user-given path or a default from the module or "action_lstm.pth".
        else:
            if debug: print("[model] lstm_model.load_model not found") # Optionally tell the user in debug mode.
            return None, None, None # Return failure (no model available).

        class_names = getattr(mod, "CLASS_NAMES", None) or DEFAULT_CLASS_NAMES # Try to get CLASS_NAMES from the module; otherwise use the backup list.
        predict_fn = getattr(mod, "predict", None) # Try to get the predict function.
        preprocess_fn = getattr(mod, "preprocess_window", None) # Try to get the preprocess_window function.
        if not callable(predict_fn) or not callable(preprocess_fn): # If either is missing…
            if debug: print("[model] lstm_model missing predict/preprocess_window") # Say so in debug.
            return None, None, None

        if debug: # If debug mode is on…
            print(f"[model] loaded weights: {weights_path or getattr(mod,'DEFAULT_WEIGHTS','action_lstm.pth')}")
            print(f"[model] CLASS_NAMES: {class_names}")

        return predict_fn, preprocess_fn, class_names # Success: return the two functions and the labels.
    except Exception as e: # If importing failed for any reason…
        if debug:
            print(f"[model] import failed: {e.__class__.__name__}: {e}")
        return None, None, None

# =========================
# Heuristic fall score (0..1): downward hip + torso tilt
# =========================
def heuristic_fall_score(seq_ntu, window=12, down_thresh=0.08): # Define a function: look at a sequence of skeletons and output a number 0..1 for “looks like a fall”.
    if len(seq_ntu) < 10: # If we don’t have enough frames…
        return 0.0 # Return “not a fall”.
    arr = np.stack(seq_ntu, axis=0) # Turn the list of skeletons into one 3-D array [time, joints, coords].
    hip_y = arr[:, 0, 1] # Take the y (vertical) position of the hip joint over time.
    L_SH, R_SH = arr[:, 4, :], arr[:, 8, :] # Pick left and right shoulder joints over time (per your joint indexing).
    mid_sh = 0.5 * (L_SH + R_SH) # Compute the mid-shoulder position each frame.
    mid_hip = arr[:, 0, :] # Grab the mid-hip position each frame.
    torso = mid_sh - mid_hip # Torso vector per frame (shoulders minus hips).
    T = min(window, len(arr)) # Only look at the last window frames.
    dy = np.diff(hip_y[-T:]) # Frame-to-frame vertical changes of the hip (recent segment).
    downward = np.clip(dy, 0, None).sum() # Keep only “moving down” amounts and sum them (how much the hip dropped).
    vx, vy = torso[-1, 0], torso[-1, 1] # Take the latest torso x and y components.
    norm = (vx**2 + vy**2) ** 0.5 + 1e-6 # Length of that vector (plus tiny number to avoid divide-by-zero).
    cos_theta = (-vy) / norm # Compute an “upward” direction cosine; using -vy makes downward tilt reduce this value.
    tilt = np.degrees(np.arccos(np.clip(cos_theta, -1, 1))) # Convert to an angle in degrees (bigger angle = more horizontal body).
    down_score = np.clip((downward - down_thresh) / 0.12, 0, 1) # Turn the drop amount into a 0..1 score (below threshold → 0; big drop → near 1).
    tilt_score = np.clip((tilt - 40) / 30, 0, 1) # Turn body tilt into a 0..1 score (upright ~0; very tilted ~1).
    return float(0.6 * down_score + 0.4 * tilt_score) # Mix them (60% drop, 40% tilt) for one “fall likelihood” number.

# =========================
# HUD drawing
# =========================
def draw_status_banner(img, label, alpha=0.8): # This function will draw a colored bar at the top of the image (img) that says the current label (“WALKING” or “FALL”). alpha=0.8 controls transparency.
    h, w = img.shape[:2] # Get the image’s height (h) and width (w) so we know how big to draw the banner and where to place text.
    bar_h = max(40, h // 18) # Decide the height of the top bar. Make it at least 40 pixels tall, or bigger if the image is large (h // 18).
    overlay = img.copy() # Make a copy of the image to draw on first; we’ll blend it back for a smooth, semi-transparent bar.
    lab = str(label).lower() # Turn the label into lowercase text so we can compare it easily (e.g., “FALL”, “fall”, “Fall” all become “fall”).
    if lab in {"fall","falling"}: # If the label indicates a fall…
        color = (0, 0, 255); text  = "FALL" # Choose red for the bar (BGR = (0,0,255)) and set the text to exactly “FALL”.
    elif lab in {"walk","walking","stand"}: # Otherwise if the label means safe/normal movement…
        color = (0, 200, 0); text  = "WALKING" # Choose green for the bar and write “WALKING”.
    else: # If it’s some other label…
        color = (50, 50, 50); text  = label.upper() # Use a gray bar and display the label in uppercase letters.
    cv2.rectangle(overlay, (0, 0), (w, bar_h), color, -1) # Paint a solid colored rectangle at the top of the overlay image: from the top-left (0,0) to (w, bar_h).
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img) # Blend the overlay with the original image, using alpha (0.8) for transparency. Result goes into img. This makes the bar look smooth and semi-transparent.
    scale = 0.9; thickness = 2 # Choose font size (scale) and line thickness for the text.
    (tw, th), baseline = cv2.getTextSize(text, TEXT_FONT, scale, thickness) # Ask OpenCV how wide (tw) and tall (th) the text will be so we can center it nicely.
    tx = (w - tw) // 2; ty = (bar_h + th) // 2 # Compute where to put the text: centered horizontally (tx) and vertically inside the banner (ty).
    cv2.putText(img, text, (tx, ty), TEXT_FONT, scale, (255,255,255), thickness, cv2.LINE_AA) # Draw the text (white color) onto the image at the computed position.
    return img # Return the modified image with the status banner drawn.

# =========================
# CSV logging helper
# =========================
def open_logger(csv_path, video_path, fps): # a function to open/create a CSV log file and write header information. Returns the open file handle, a CSV writer, and the final path.
    if csv_path is None or csv_path == "": # If the user didn’t specify a CSV path…
        ts = datetime.now().strftime("%Y%m%d_%H%M%S") # Make a timestamp string like 20250830_134522 for uniqueness.
        os.makedirs("logs", exist_ok=True) # Make a logs/ folder if it doesn’t exist yet.
        base = os.path.splitext(os.path.basename(video_path))[0] # Take the video filename (without extension) to include in the log name.
        csv_path = os.path.join("logs", f"{base}_{ts}.csv") # Build a default path like logs/myvideo_20250830_134522.csv.

    parent = os.path.dirname(csv_path) # Get the parent folder of the CSV file (in case the user passed a nested path).
    if parent and not os.path.exists(parent): # If that folder doesn’t exist…
        os.makedirs(parent, exist_ok=True) # Create it.

    f = open(csv_path, "w", newline="") # Open the CSV file for writing ("w"). newline="" avoids extra blank lines on Windows.
    w = csv.writer(f) # Make a CSV writer object so we can write rows easily.

    # Write a tiny header with config for reproducibility (as comments)
    w.writerow([f"# video={os.path.basename(video_path)}", # Start writing a single row (list) with comment fields. First item: which video file.
                f"fps={fps:.3f}", # Second item in the header row: the frames-per-second, formatted nicely.
                f"FRAMES_PER_WINDOW={FRAMES_PER_WINDOW}", # Include key settings so you remember how you ran it.
                f"SMOOTH_WINDOW={SMOOTH_WINDOW}", # Another config value in the header.
                f"ENTRY_DWELL={ENTRY_DWELL}", # Include dwell setting in the header.
                f"FALL_PROB_ON={FALL_PROB_ON}", # Hysteresis ON threshold.
                f"FALL_PROB_OFF={FALL_PROB_OFF}", # Hysteresis OFF threshold.
                f"USE_HEURISTIC_FUSION={USE_HEURISTIC_FUSION}", # Whether fusion is on.
                f"HEURISTIC_WEIGHT={HEURISTIC_WEIGHT}", # Blend weight.
                f"HEURISTIC_FORCE_ON={HEURISTIC_FORCE_ON}", # Force-on threshold.
                f"DUAL_GATE={DUAL_GATE}", # Whether dual gate is used.
                f"MODEL_GATE={MODEL_GATE}", # Model gate value.
                f"HEUR_GATE={HEUR_GATE}"]) # Heuristic gate value (close the header row list).
    # Column header
    w.writerow(["frame_idx","time_sec", # Start the column header row. First columns: frame number and time (seconds).
                "model_prob_fall","heuristic_score", # Columns for model fall probability and heuristic score.
                "fused_prob","avg_prob","state"]) # Columns for blended probability, smoothed average probability, and state (“walk/fall”).
    f.flush() # Push data to disk so the file isn’t empty if the program stops early.
    return f, w, csv_path # Return the file handle, writer, and final CSV path to the caller.

# =========================
# Core: continuous playback with status overlay + debug
# =========================
def process_one(video_path, use_model=False, weights_path=None, debug=False, log_csv=None):
    # The main function process_one. It processes one video with all features. Options: use the model or not, which weights to load, debug prints, and whether to log to CSV.
    cap = cv2.VideoCapture(video_path) # Open the video file so we can read frames from it.
    if not cap.isOpened():
        print(f"[error] Could not open: {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Ask how many frames are in the video (estimate; sometimes this is 0 or not exact).
    fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 30 # Ask the frames per second. If it’s missing/0, default to 30.
    if fps <= 0: fps = 30 # Double-check and force 30 if FPS is invalid.
    delay_ms = max(1, int(1000.0 / fps)) # Compute how long to wait between frames (in milliseconds) for natural-speed playback.

    print(f"[info] Playing {video_path} | frames≈{total} | fps≈{fps:.2f}") # Print basic info so you know what’s going on.

    if not has_gui(): # If this machine can’t show windows…
        print("[warn] No GUI backend; install 'opencv-python' (not -headless).")
        return

    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL) # Create a window named “Preview” that we can resize.
    try: cv2.setWindowProperty("Preview", cv2.WND_PROP_TOPMOST, 1) # Try to make the window stay on top (some systems ignore this).
    except Exception: pass # If that property call failed, silently ignore it.

    # Try to load model (if requested)
    predict_fn = preprocess_fn = None # Initialize model function handles to None.
    class_names = list(DEFAULT_CLASS_NAMES) # Start with default class names; we’ll override if the model provides its own.
    if use_model: # If the user asked to use the model…
        predict_fn, preprocess_fn, class_names = try_load_predictor(weights_path=weights_path, debug=debug) # Try to load the model, its preprocessor, and class names.
        model_available = predict_fn is not None and preprocess_fn is not None # Check if both pieces arrived successfully.
        if not model_available:
            print("[warn] Model requested but not available; using heuristic.")
        use_model = model_available

    # CSV logger
    log_file = writer = None # Initialize CSV log variables.
    if log_csv is not None and log_csv != "": # If the user wants logging…
        log_file, writer, log_csv = open_logger(log_csv, video_path, fps) # Open the log file (creating default path if necessary) and get a CSV writer.
        print(f"[log] Writing CSV to: {os.path.abspath(log_csv)}") # Tell the user where the log is being written.

    frames = deque(maxlen=FRAMES_PER_WINDOW) # Make a sliding buffer to hold the most recent frames (images).
    skels  = deque(maxlen=FRAMES_PER_WINDOW)      # original coords for HUD + heuristic. Make a sliding buffer for the most recent skeletons (original coordinates).
    skels_norm = deque(maxlen=FRAMES_PER_WINDOW)  # normalized coords for the model. Make a sliding buffer for normalized skeletons (hips-centered, torso-scaled) for the model.
    frame_idx = 0 # Counter for which frame number we’re on.
    paused = False # Start in “playing” mode. Space bar will toggle this.

    # state/hysteresis
    state_is_fall = False # Current state: False means WALKING; True would mean FALL.
    smooth_probs = deque(maxlen=SMOOTH_WINDOW) # Buffer to hold recent probabilities for smoothing.
    fall_entry_count = 0 # How many frames in a row we’ve been “triggered” (for dwell).

    while True: # Start the main loop: keep running until we break (end of video or quit).
        if not paused: # Only grab a new frame if we’re not paused.
            ret, frame = cap.read() # Read the next frame from the video. ret is True if it worked.
            if not ret: # If there are no more frames (or read failed)…
                break
            frame_idx += 1 # Increase the frame counter.
            skel = extract_skeleton(frame) # Run pose detection on this frame; get the 25×3 skeleton (or zeros).
            frames.append(frame) # Store the frame in our sliding window.
            skels.append(skel) # Store the raw skeleton in the sliding window.
            # NEW: store normalized skeleton alongside
            skels_norm.append(normalize_skeleton(skel)) # Compute and store the normalized version of the skeleton.

        disp = frames[-1].copy() if frames else np.zeros((480, 640, 3), dtype=np.uint8)
        # Make an image to display (disp). If we have frames, copy the latest one; otherwise, make a blank 480×640 black image.
        if frames:
            disp = draw_points(disp, skels[-1]) # Draw green dots for the latest skeleton on the display image.

        # per-window probabilities
        model_prob_fall = 0.0 # Initialize model fall probability to 0.
        fused_prob = 0.0 # Initialize blended probability to 0.
        h_score = 0.0 # Initialize heuristic score to 0.

        if len(skels) == FRAMES_PER_WINDOW: # Only do window-based processing when we actually have a full window (e.g., 45 frames).
            if use_model: # If we have a model loaded and we’re supposed to use it…
                try:
                    # feed the NORMALIZED sequence to the model
                    x = preprocess_fn(list(skels_norm))   # -> [1, T, 150]. Preprocess the normalized skeleton window into the exact tensor shape the model expects (often batch=1, time=T, features=25*3=75 or flattened 150, etc. per your design).
                    probs = predict_fn(x)                 # -> [1, C]. Run the model to get class probabilities (C classes) for this window.
                    probs = np.array(probs).reshape(-1) # Turn the output into a flat 1-D NumPy array (e.g., [p_walk, p_fall, ...]).
                    # find fall index
                    names_lower = [str(n).lower() for n in class_names] # Lowercase the class names list so we can match “fall” reliably.
                    if "fall" in names_lower: # If “fall” exists in the class names list…
                        fall_idx = names_lower.index("fall") # Find the index (position) where “fall” lives.
                        model_prob_fall = float(probs[fall_idx]) if fall_idx < probs.size else float(probs.max()) # Use that exact probability for “fall” if the index is valid; otherwise, fallback to the maximum probability.
                    else:
                        model_prob_fall = float(probs.max()) # Use the maximum probability as a best-effort fallback.
                except Exception as e:
                    if debug:
                        print(f"[debug] model call failed: {e}")
                    model_prob_fall = 0.0 # Set model fall probability to 0 on failure.

            if USE_HEURISTIC_FUSION:
                # heuristic uses original coords (keeps your tuned thresholds intact)
                h_score = heuristic_fall_score(list(skels)) # Compute the heuristic fall score from the window of skeletons.
                fused_prob = float((1 - HEURISTIC_WEIGHT) * model_prob_fall + HEURISTIC_WEIGHT * h_score) # Blend the model probability with the heuristic using the set weight (e.g., 66% model + 34% heuristic).
            else:
                fused_prob = model_prob_fall

        # ----- smoothing -----
        fall_prob = fused_prob # Take the current fused probability as the “fall prob” for this step.
        smooth_probs.append(fall_prob) # Add it into the smoothing deque (oldest value falls off automatically).
        avg_prob = float(np.mean(smooth_probs)) if len(smooth_probs) else fall_prob # Compute the average over the recent values (or just the current value if buffer is empty).

        # ===== DECISION LOGIC (conservative, debounced, dual-gated) =====
        trigger = False

        # Base trigger from smoothed model prob
        if avg_prob >= FALL_PROB_ON: # If the smoothed probability is high enough…
            trigger = True

        # Heuristic force-on (only very strong)
        if h_score >= HEURISTIC_FORCE_ON: # If the heuristic is very confident by itself…
            trigger = True

        # Require BOTH model and heuristic to be decent (conservative)
        if DUAL_GATE:
            trigger = (avg_prob >= MODEL_GATE) and (h_score >= HEUR_GATE) # Only set trigger if both smoothed model prob and heuristic score beat their gates.

        # Debounce / dwell: need sustained trigger for ENTRY_DWELL frames
        if trigger:
            fall_entry_count += 1 # Increase the counter for “how many frames in a row it’s been true”.
        else:
            fall_entry_count = 0

        # Apply state transition with hysteresis
        if not state_is_fall: # If we’re currently in WALK state…
            if fall_entry_count >= ENTRY_DWELL: # If we’ve had a sustained trigger long enough (dwell)…
                state_is_fall = True
        else:
            if avg_prob < FALL_PROB_OFF: # Only switch back when the smoothed probability drops below the OFF threshold.
                state_is_fall = False

        label_smoothed = "fall" if state_is_fall else "walk"

        # HUD + diagnostics
        if HUD:
            disp = draw_status_banner(disp, label_smoothed) # Draw a red “FALL” banner if state_is_fall is True, else green “WALKING”.
            txt = (f"Frame {frame_idx}/{total}  model={model_prob_fall:.2f}  "
                   f"fused={fall_prob:.2f}  avg={avg_prob:.2f}  h={h_score:.2f}  "
                   f"(model={'on' if use_model else 'off'})  space=pause q=quit")
            cv2.putText(disp, txt, (12, max(48, disp.shape[0]//18)+28),
                        TEXT_FONT, 0.6, (220,220,220), 2, cv2.LINE_AA)

        # CSV logging (per frame)
        if writer is not None: # If CSV logging is enabled…
            t_sec = frame_idx / float(fps) if fps > 0 else 0.0 # Compute the current time in seconds.
            writer.writerow([ # Write a row with frame index, time, model fall probability, and heuristic score…
                frame_idx,
                f"{t_sec:.3f}",
                f"{model_prob_fall:.6f}",
                f"{h_score:.6f}",
                f"{fall_prob:.6f}",
                f"{avg_prob:.6f}",
                "FALL" if state_is_fall else "WALK"
            ])
            # keep the file up to date in case of crash
            if frame_idx % 30 == 0:
                log_file.flush()

        cv2.imshow("Preview", disp) # Show the display image (with dots and banner) in the “Preview” window.
        if debug and len(skels) == FRAMES_PER_WINDOW:
            print(f"[debug] f={frame_idx:5d} model={model_prob_fall:.3f} fused={fall_prob:.3f} "
                  f"avg={avg_prob:.3f} h={h_score:.3f} state={'FALL' if state_is_fall else 'WALK'} "
                  f"(model={'on' if use_model else 'off'})")

        k = cv2.waitKey(0 if paused else delay_ms) & 0xFF # Wait for a key press briefly. If paused, wait forever (0) until you press something; otherwise wait delay_ms. & 0xFF standard way to get the low byte.
        if k in (ord('q'), 27): # If you press q or ESC (27)…
            break
        elif k == ord(' '): # If you press the Space bar (ASCII 32)…
            paused = not paused

    cap.release() # Close the video file.
    if log_file is not None:
        log_file.flush()
        log_file.close()
    cv2.destroyAllWindows() # Close any OpenCV windows we opened.
    print(f"[done] {video_path} | played {frame_idx} frames")

# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser() # Create an argument parser that will understand command-line options, with a human description.
    # following are the different options - self-explanatory.
    ap.add_argument("--video", help="File, folder, or glob. Example: --video test3.mp4")
    ap.add_argument("--use-model", action="store_true", help="Use lstm_model (predict + preprocess).")
    ap.add_argument("--weights", default=None, help="Path to weights file (e.g., action_lstm.pth).")
    ap.add_argument("--debug", action="store_true", help="Print per-frame diagnostics.")
    ap.add_argument("--log-csv", default="", help="CSV file to log per-frame metrics. If omitted, saves to logs/<video>_<ts>.csv")
    args = ap.parse_args() # Read the actual options the user typed when running the script.

    any_processed = False
    for vp in iter_video_paths(args.video): # Use our earlier helper to yield every matching video path (folder/file/pattern).
        process_one(vp, use_model=args.use_model, weights_path=args.weights, debug=args.debug, log_csv=args.log_csv) # Run the full pipeline on this file: open, pose, model+heuristic, HUD, logging, keyboard controls.
        any_processed = True # Mark that we have something to play.
    if not any_processed:
        print("[error] No matching videos found. Use --video <path> or put test3.mp4 in the working directory.")

if __name__ == "__main__": # This means “only run the following code if you start this file directly (not if it’s imported by another file).” This is the command-line interface (CLI) part.
    main()
