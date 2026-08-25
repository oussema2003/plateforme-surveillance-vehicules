from ultralytics import YOLO

model = YOLO("/teamspace/studios/this_studio/car_brand_detector/runs/detect/car_logos_gbnlb/stage1-2/weights/best.pt")
metrics = model.val(
    data="dataset_gbnlb/data.yaml",
    split="test",
    imgsz=640,
    conf=0.25,
    iou=0.6,
    plots=True,
)
print(f"\nmAP50 (test)    : {metrics.box.map50:.4f}")
print(f"mAP50-95 (test) : {metrics.box.map:.4f}")

print("\nmAP50 par classe :")
for i, name in enumerate(model.names.values()):
    try:
        print(f"  {name}: {metrics.box.maps[i]:.4f}")
    except IndexError:
        pass