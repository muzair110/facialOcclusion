import cv2
from ultralytics import YOLO
import numpy as np
import warnings
import tempfile
import os

def detectOcclusion(imagePath):
    image = cv2.imread(imagePath)
    if image is None:
        print(f"Error: Could not read image from {imagePath}")
        return None
   
    imageRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Map for the trained classes list
    CLASS_NAMES = {
        0: 'cloth', 1: 'ear_r', 2: 'eye_g', 3: 'hair', 4: 'hat',
        5: 'l_brow', 6: 'l_ear', 7: 'l_eye', 8: 'l_lip', 9: 'mouth',
        10: 'neck', 11: 'neck_l', 12: 'nose', 13: 'r_brow', 14: 'r_ear',
        15: 'r_eye', 16: 'skin', 17: 'u_lip'
    }

    # Suppress warnings
    warnings.filterwarnings('ignore')
   
    # Load your segmentation model
    model = YOLO("faceParsing_int8.tflite", task='segment')
   
    try:
        results = model.predict(source=imagePath, conf=0.05, iou=0.45, max_det=20, verbose=False)[0]
    except KeyError:
        results = model(imageRgb, verbose=False)[0]

    CRITICAL_FEATURES = {
        'left_eye': [7],
        'right_eye': [15],
        'nose': [12],
        'lips': [17, 8]
    }
   
    MOUTH_OPEN_CLASS = 9
    EYE_GLASSES_CLASS = 2
 
    occlusion_report = {
        'is_valid': True,
        'rejection_reason': None,
        'is_occluded': False,
        'occluded_features': [],
        'detected_features': [],
        'has_glasses': False,
        'mouth_open': False,
        'confidence_scores': {},
        'all_detected_classes': []
    }
 
    if results is not None and len(results.boxes) > 0:
        detectedClasses = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
       
        detected_class_ids = detectedClasses.astype(int).tolist()
        valid_class_ids = {cls_id for cls_id in detected_class_ids if 0 <= cls_id <= 17}
       
        for cls_id in valid_class_ids:
            if cls_id in CLASS_NAMES:
                occlusion_report['all_detected_classes'].append(CLASS_NAMES[cls_id])
       
        if MOUTH_OPEN_CLASS in valid_class_ids:
            occlusion_report['mouth_open'] = True
            occlusion_report['is_valid'] = False
            occlusion_report['rejection_reason'] = "Mouth is open"
       
        if EYE_GLASSES_CLASS in valid_class_ids:
            occlusion_report['has_glasses'] = True
       
        for feature_name, class_ids in CRITICAL_FEATURES.items():
            feature_detected = any(cls_id in valid_class_ids for cls_id in class_ids)
           
            if feature_detected:
                occlusion_report['detected_features'].append(feature_name)
                for i, cls_id in enumerate(detectedClasses):
                    if int(cls_id) in class_ids:
                        occlusion_report['confidence_scores'][feature_name] = float(confidences[i])
                        break
            else:
                occlusion_report['occluded_features'].append(feature_name)
                occlusion_report['is_occluded'] = True
       
        if occlusion_report['is_occluded'] and not occlusion_report['rejection_reason']:
            occlusion_report['is_valid'] = False
            occlusion_report['rejection_reason'] = "Face features occluded"
       
        print("\n=== FACE OCCLUSION DETECTION REPORT ===")
        print(f"Image: {imagePath}")
        print(f"Detected: {occlusion_report['detected_features']}")
        print(f"Occluded: {occlusion_report['occluded_features']}")
        if occlusion_report['mouth_open']:
            print("❌ Rejected: Mouth open")
        elif occlusion_report['is_occluded']:
            print("❌ Rejected: Features occluded")
        else:
            print("✅ Accepted: All good")
       
    else:
        print("No facial features detected")
        occlusion_report['is_valid'] = False
        occlusion_report['rejection_reason'] = "No facial features detected"
        occlusion_report['is_occluded'] = True
        occlusion_report['occluded_features'] = list(CRITICAL_FEATURES.keys())
   
    return occlusion_report


if __name__ == '__main__':
    # --- Capture from webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    print("Press SPACE to capture, ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Webcam - Press SPACE to capture", frame)

        key = cv2.waitKey(1)
        if key % 256 == 27:   # ESC pressed
            print("Escape hit, closing...")
            break
        elif key % 256 == 32: # SPACE pressed
            # Save frame to a temp file
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(tmp_file.name, frame)
            print(f"Captured {tmp_file.name}")
            cap.release()
            cv2.destroyAllWindows()

            # Run detection on the captured image
            result = detectOcclusion(tmp_file.name)
            print("\nFinal Result:", result)

            # Cleanup
            os.unlink(tmp_file.name)
            break

    cap.release()
    cv2.destroyAllWindows()
