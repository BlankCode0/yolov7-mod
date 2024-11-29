from models.yolo import Model

cus_model = Model(cfg=".//cfg//training//yolov7-mod.yaml")
print(cus_model)


yolo_model = Model(cfg=".//cfg//training//yolov7.yaml")
print(yolo_model)
