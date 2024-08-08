from typing import Optional

import gradio as gr
import numpy as np
import torch
from PIL import Image
import cv2
import os
import imageio

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from utils.models import (
    load_models,
    CHECKPOINT_NAMES,
)

MARKDOWN = """
# Segment Anything Model 2 Video
"""
DEVICE = torch.device("cuda")
VIDEO_GENERATORS = load_models()


def get_meta_from_video(checkpoint_dropdown, input_video):
    if input_video is None:
        return None, None, None, None

    # capture the first frame
    cap = cv2.VideoCapture(input_video)

    vedio_name = input_video.split("/")[-1].split(".")[0]
    output_dir = f"files/images/{vedio_name}"
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if frame_number == 0:
            first_frame = frame
        if not ret:
            break
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"{frame_number:04d}.jpg")
        cv2.imwrite(output_filename, frame)
        frame_number += 1
    cap.release()

    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    # set inference state
    predictor = VIDEO_GENERATORS[checkpoint_dropdown]
    inference_state = predictor.init_state(video_path=output_dir)

    return first_frame, first_frame, predictor, inference_state


def get_click_prompt(click_stack, point):
    click_stack[0].append(point["coord"])
    click_stack[1].append(point["mode"])

    prompt = {
        "points_coord": click_stack[0],
        "points_mode": click_stack[1],
        "multimask": "True",
    }

    return prompt


def add_mask_to_image(image, mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # Orange color with 60% opacity
        color = np.array([255 / 255, 165 / 255, 0 / 255, 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    # Make sure the image is in the right shape
    if image.shape[-1] != 3:
        raise ValueError("The input image must have 3 color channels.")

    # Blend the mask with the image
    for c in range(3):
        image[:, :, c] = (1 - mask_image[:, :, 3]) * image[:, :, c] + mask_image[
            :, :, 3
        ] * mask_image[:, :, c] * 255

    return image.astype(np.uint8)


def draw_circles_on_image(image, coords, labels, radius=5):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    for point in pos_points:
        cv2.circle(image, (int(point[0]), int(point[1])), radius, (0, 255, 0), -1)

    for point in neg_points:
        cv2.circle(image, (int(point[0]), int(point[1])), radius, (255, 0, 0), -1)

    return image


def seg(predictor, inference_state, click_prompt, origin_frame):
    points = np.array(click_prompt["points_coord"], dtype=np.float32)
    labels = np.array(click_prompt["points_mode"], np.int32)

    ann_frame_idx = 0
    ann_obj_id = 1
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, _, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

    origin_frame = draw_circles_on_image(origin_frame, points, labels)
    masked_frame = add_mask_to_image(
        origin_frame,
        (out_mask_logits[0] > 0.0).cpu().numpy(),
    )
    return masked_frame


def click(
    predictor,
    inference_state,
    origin_frame,
    point_mode,
    click_stack,
    evt: gr.SelectData,
):
    if point_mode == "Positive":
        point = {"coord": [evt.index[0], evt.index[1]], "mode": 1}
    else:
        point = {"coord": [evt.index[0], evt.index[1]], "mode": 0}

    # get click prompts for sam to predict mask
    origin_frame = origin_frame.copy()
    click_prompt = get_click_prompt(click_stack, point)
    masked_frame = seg(predictor, inference_state, click_prompt, origin_frame)

    return masked_frame, click_stack


def seg_video(predictor, inference_state, video_path):
    video_segments = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    # render the segmentation results every few frames
    video_name = video_path.split("/")[-1].split(".")[0]
    images_dir = f"files/images/{video_name}"
    frame_names = sorted(os.listdir(images_dir))
    output_dir = f"files/masks/{video_name}"
    os.makedirs(output_dir, exist_ok=True)
    masked_frames_dir = f"files/masked_frames/{video_name}"
    os.makedirs(masked_frames_dir, exist_ok=True)

    vis_frame_stride = 1
    masked_frames = []

    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            # save pure mask
            if out_obj_id != 1:
                print(f"Skipping object id {out_obj_id}...")
                continue
            mask_filename = os.path.join(output_dir, f"{out_frame_idx:04d}.png")
            mask_2d = out_mask.squeeze()
            mask_int = np.where(mask_2d, 255, 0).astype(np.uint8)
            image = Image.fromarray(mask_int)
            image.save(mask_filename)
            # load frame to np array
            frame_filename = os.path.join(images_dir, frame_names[out_frame_idx])
            frame = cv2.imread(frame_filename)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # add mask to frame
            masked_frame = add_mask_to_image(frame, out_mask)
            masked_frames.append(masked_frame)

    # save preview video
    print(f'Saving video "{video_name}"...')
    os.makedirs("files/videos", exist_ok=True)
    video_filename = f"files/videos/{video_name}.mp4"

    writer = imageio.get_writer(video_filename, fps=30, codec='libx264')
    for frame in masked_frames:
        writer.append_data(frame)
    writer.close()

    return video_filename


def reset(origin_frame):
    return [[], []], origin_frame


with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)

    click_stack = gr.State([[], []])
    predictor = gr.State(None)
    inference_state = gr.State(None)
    origin_frame = gr.State(None)

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Upload video")
        with gr.Column():
            image_prompter_input_component = gr.Image(
                label="First frame", interactive=True
            )

    with gr.Row():
        with gr.Column():
            segmented_video = gr.Video(label="Segmented video")
        with gr.Column():
            with gr.Row():
                checkpoint_dropdown_component = gr.Dropdown(
                    choices=CHECKPOINT_NAMES,
                    value=CHECKPOINT_NAMES[3],
                    label="Checkpoint",
                    info="Select a SAM2 checkpoint to use.",
                    interactive=True,
                )
                point_mode = gr.Radio(
                    choices=["Positive", "Negative"],
                    value="Positive",
                    label="Point prompt type",
                    interactive=True,
                )
            with gr.Row():
                click_reset_but = gr.Button(value="reset", interactive=True)
                click_submit_but = gr.Button(value="submit", interactive=True)

    input_video.change(
        fn=get_meta_from_video,
        inputs=[checkpoint_dropdown_component, input_video],
        outputs=[
            image_prompter_input_component,
            origin_frame,
            predictor,
            inference_state,
        ],
    )

    image_prompter_input_component.select(
        fn=click,
        inputs=[predictor, inference_state, origin_frame, point_mode, click_stack],
        outputs=[image_prompter_input_component, click_stack],
    )

    click_submit_but.click(
        fn=seg_video,
        inputs=[
            predictor,
            inference_state,
            input_video,
        ],
        outputs=[segmented_video],
    )

    click_reset_but.click(
        fn=reset,
        inputs=[origin_frame],
        outputs=[click_stack, image_prompter_input_component],
    )

demo.launch(debug=True, show_error=True)
