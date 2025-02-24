import cv2
from ultralytics import YOLO

def detect_people(video_path, model, trashhold=0.5) :
    """
    Из исходного видеоролика создает видео, в котором обнаруженные люди обведены в рамку
    с указанием уверенности в обнаружении.

    Параметры:
    video_path (string): Абсолютный путь к файлу.
    model (YOLO): Модель для обнаружения обьектов.
    trashhold (float): Порог уверенности, при котором обьект будет выделен в рамку.
    По умолчанию равен 0.5.

    Возвращает:
    None
    """
    
    #Открытие видеофайла
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    #Создание нового видео
    name = video_path.split('\\')[-1].split('.')[0]
    output_path = f'{name}_detected.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    color = (0, 255, 0)#зеленая рамка
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    #Обработка каждого кадра видео
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)

        #Отрисовка рамки для каждого обнаруженного объекта
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                #Класс 0 соответствует людям
                if box.cls[0] == 0:
                    confidence = round(box.conf[0], 2)
                    #Проверка порогового значения уверенности
                    if confidence > trashhold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_name = "Person"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        label = f"{class_name} {confidence}"
                        (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        label_x = x1
                        label_y = y1 - 10 if y1 > 10 else y1 + label_height + 10
                        cv2.putText(frame, label, (label_x, label_y), font, font_scale, color, thickness)

        # Запись обработанного кадра в выходной файл
        out.write(frame)

    # Освобождение ресурсов
    cap.release()
    out.release()
    print(f'Поиск людей на видео завершен. Результат сохранен в {output_path}')

if __name__ == '__main__':
    #Загрузка YOLOv8
    model = YOLO('yolov8n.pt') 
    detect_people('crowd.mp4', model)