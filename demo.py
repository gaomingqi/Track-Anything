from metaseg import SegAutoMaskPredictor, SegManualMaskPredictor, SahiAutoSegmentation, sahi_sliced_predict

# For image

def automask_image_app(image_path, model_type, points_per_side, points_per_batch, min_area):
    SegAutoMaskPredictor().image_predict(
        source=image_path,
        model_type=model_type,  # vit_l, vit_h, vit_b
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        min_area=min_area,
        output_path="output.png",
        show=False,
        save=True,
    )
    return "output.png"


# For video

def automask_video_app(video_path, model_type, points_per_side, points_per_batch, min_area):
    SegAutoMaskPredictor().video_predict(
        source=video_path,
        model_type=model_type,  # vit_l, vit_h, vit_b
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        min_area=min_area,
        output_path="output.mp4",
    )
    return "output.mp4"


# For manuel box and point selection

def manual_app(image_path, model_type, input_point, input_label, input_box, multimask_output, random_color):
    SegManualMaskPredictor().image_predict(
        source=image_path,
        model_type=model_type,  # vit_l, vit_h, vit_b
        input_point=input_point,
        input_label=input_label,
        input_box=input_box,
        multimask_output=multimask_output,
        random_color=random_color,
        output_path="output.png",
        show=False,
        save=True,
    )
    return "output.png"


# For sahi sliced prediction

def sahi_autoseg_app(
    image_path,
    sam_model_type,
    detection_model_type,
    detection_model_path,
    conf_th,
    image_size,
    slice_height,
    slice_width,
    overlap_height_ratio,
    overlap_width_ratio,
):
    boxes = sahi_sliced_predict(
        image_path=image_path,
        detection_model_type=detection_model_type,  # yolov8, detectron2, mmdetection, torchvision
        detection_model_path=detection_model_path,
        conf_th=conf_th,
        image_size=image_size,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    SahiAutoSegmentation().predict(
        source=image_path,
        model_type=sam_model_type,
        input_box=boxes,
        multimask_output=False,
        random_color=False,
        show=False,
        save=True,
    )
    
    return "output.png"
