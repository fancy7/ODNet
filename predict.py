#-----------------------------------------------------------------------#
#   The predict.py file integrates various functions such as image prediction, camera detection, FPS testing, and directory traversal detection into a single Python file, which can be modified by specifying a mode.
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    # mode is used to specify the testing mode:
    # 'predict' indicates single image prediction. If you want to modify the prediction process, such as saving the image or cropping the object, please refer to the detailed comments below.
    # 'video' indicates video detection, which can be used to detect objects in a camera or video. For more information, please refer to the comments below.
    # 'fps' indicates testing fps. The image used is street.jpg in the img folder. For more information, please refer to the comments below.
    # 'dir_predict' indicates detecting objects in a folder and saving the results. The default folder to be traversed is the img folder, and the results are saved in the img_out folder. For more information, please refer to the comments below.
    # 'heatmap' indicates visualizing the prediction results as a heatmap. For more information, please refer to the comments below.
    # 'export_onnx' indicates exporting the model to ONNX format, which requires PyTorch 1.7.1 or above.
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_heatmap"
    #-------------------------------------------------------------------------#
    # crop specifies whether to crop the target after predicting on a single image
    # count specifies whether to count the targets
    # crop and count are only valid when mode='predict'
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    # video_path is used to specify the path of the video. When video_path is 0, it means that the camera is being detected.
    # If you want to detect a video, set it to "xxx.mp4" for example, which represents reading the xxx.mp4 file in the root directory.
    # video_save_path indicates the path where the video will be saved. When video_save_path is "", it means that the video will not be saved.
    # If you want to save the video, set it to "yyy.mp4" for example, which represents saving the file as yyy.mp4 in the root directory.
    # video_fps is used for the fps of the saved video. #
    # video_path, video_save_path, and video_fps are only valid when mode='video'
    # Saving the video requires ctrl+c to exit or running until the last frame to complete the full saving process.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    # test_interval is used to specify the number of image detection times for measuring fps. In theory, the larger the test_interval, the more accurate the fps will be.
    # fps_image_path is used to specify the path of the fps test image. #
    # The `test_interval` and `fps_image_path` options are only valid when `mode='fps'`.
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 5000
    fps_image_path  = "img/20001.jpg"
    #-------------------------------------------------------------------------#
    # dir_origin_path specifies the folder path of the image used for detection
    # dir_save_path specifies the path where the detected images will be saved #
    # dir_origin_path and dir_save_path are only valid when mode='dir_predict'
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    # heatmap_save_path:  The path to save the heatmap, which is saved by default in the model_data folder. #
    # The `heatmap_save_path` is only valid when `mode` is set to 'heatmap'.
    #-------------------------------------------------------------------------#
    heatmap_save_path = "heatmap/"
    #-------------------------------------------------------------------------#
    #   simplify            Use Simplify onnx
    #   onnx_save_path      The path for saving the ONNX model has been specified.
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to read the camera (video) correctly. Please make sure that the camera is properly installed and that the video path is correctly filled in.")

        fps = 0.0
        while(True):
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(yolo.detect_image(frame))
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()
        
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg",".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)

    elif mode == "dir_heatmap":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                heatmap_save_path = os.path.join(heatmap_save_path, img_name.replace(".jpg", ".png"))
                r_image = yolo.detect_heatmap(image, heatmap_save_path)
                heatmap_save_path = "heatmap/"
                if not os.path.exists(heatmap_save_path):
                    os.makedirs(heatmap_save_path)

                
    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
