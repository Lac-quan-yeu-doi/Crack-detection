import argparse
from ultralytics import YOLO
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='runs/detect/crack_train/weights/best.pt', help='Path to trained model')
    parser.add_argument('--source', type=str, default='path/to/test_image.jpg', help='Image/video path or 0 for webcam')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (0-1)')
    parser.add_argument('--save', action='store_true', help='Save output video/image')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Load trained model
    model = YOLO(args.model)
    
    # Run inference
    results = model(args.source, conf=args.conf, verbose=True)
    
    # Process results (bbox visualization)
    for r in results:
        boxes = r.boxes  # Bbox detections
        if boxes is not None:
            for box in boxes:
                # Get bbox coords
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Draw on image (if image input)
                if r.ims.shape[0] == 1:  # Single image
                    img = r.orig_img[0]
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(img, f'Crack {conf:.2f}', (int(x1), int(y1)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.imwrite('inference_result.jpg', img) if args.save else cv2.imshow('Crack Detection', img)
        
        # For video: results save to 'runs/detect/predict/'
        if args.save:
            r.save = True  # Auto-save annotated frames
    
    print("Inference complete! Check 'runs/detect/predict/' for outputs.")