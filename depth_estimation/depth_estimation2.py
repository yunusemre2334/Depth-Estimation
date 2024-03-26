import cv2
import imutils
import torch 
from ultralytics import YOLO 

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("video.mp4")

midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

transforms = torch.hub.load('intel-isl/MiDaS','transforms')
transform = transforms.small_transform

threshold = 0.3

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=500)
    # results = model(frame)[0]
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # imgbatch = transform(img).to('cpu')
    
    with torch.no_grad():
        results = model(frame)[0]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgbatch = transform(img).to('cpu')
    

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        with torch.no_grad():
            prediction = midas(imgbatch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size = img.shape[:2],
                mode = 'bicubic',
                align_corners=False

            ).squeeze()

            output = prediction.cpu().numpy()

            depth_map = cv2.normalize(prediction.cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_map_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

            if score > threshold:
                cv2.rectangle(depth_map_color, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(depth_map_color, f'{model.names[int(class_id)]}: {score:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                object_depth = output[int((y1 + y2) / 2), int((x1 + x2) / 2)]

                cv2.putText(depth_map_color, f'Depth: {object_depth:.2f}m', (int(x1), int(y1) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    combined_image = cv2.hconcat([frame, depth_map_color])
    
    cv2.imshow("Depth Map with Object Detection", combined_image)
    
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break 

cap.release()
cv2.destroyAllWindows()