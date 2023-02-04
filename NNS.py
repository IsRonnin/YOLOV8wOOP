from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont


class NN:
    def __init__(self, name_m: str = "yolov8s.pt"):
        self.__model = YOLO(name_m)

    def save(self, image_path: str):
        image = Image.open(image_path)
        self.__model.predict(source=image, classes=[0], save=True)

    def custom_save(self, image_path: str, file_name: str):
        image = Image.open(image_path)
        results = self.__model(source=image)
        boxes = results[0].boxes.xyxy
        classes = results[0].boxes.cls
        confidence = results[0].boxes.conf
        print(boxes)  # TODO Remove after debug
        print(classes)
        print(confidence)
        for i in range(len(boxes)):
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("Montserrat-Bold.ttf", 16)
            if classes[i] == 0:
                draw.text((boxes[i][0] - 25, boxes[i][1] - 25), "People", (15, 15, 15), font=font,
                          stroke_width=2, stroke_fill=(255, 255, 255))
                draw.rounded_rectangle(
                    (boxes[i][0] - 8, boxes[i][1] - 8, boxes[i][2] + 8,
                     boxes[i][3] + 8), outline=(255, 255, 255), width=2, radius=5)
                draw.rounded_rectangle(
                    (boxes[i][0] - 6, boxes[i][1] - 6, boxes[i][2] + 6,
                     boxes[i][3] + 6), outline=(0, 0, 0), width=2, radius=5)
        image.save(fp=f"{file_name}.png")
