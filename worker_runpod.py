import os, json, requests, random, runpod

import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_video_predictor
import shutil
import subprocess
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

def detect_body_keypoints(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(frame_rgb).unsqueeze(0).to('cuda')
    with torch.no_grad():
        prediction = body_detector(img_tensor)[0]
    if len(prediction['boxes']) > 0:
        best_box = prediction['boxes'][prediction['scores'].argmax()].cpu().numpy()
        x1, y1, x2, y2 = best_box
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        width, height = x2 - x1, y2 - y1
        offset_x, offset_y = width * 0.2, height * 0.2
        keypoints = np.array([
            [center_x, center_y],
            [center_x - offset_x, center_y],
            [center_x + offset_x, center_y],
            [center_x, center_y - offset_y],
            [center_x, center_y + offset_y],
        ], dtype=np.float32)
        keypoints[:, 0] = np.clip(keypoints[:, 0], x1, x2)
        keypoints[:, 1] = np.clip(keypoints[:, 1], y1, y2)
        return keypoints
    else:
        height, width = frame.shape[:2]
        center = np.array([[width // 2, height // 2]], dtype=np.float32)
        return np.tile(center, (5, 1))

def remove_background(frame, mask, bg_color):
    mask = mask.squeeze()
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    else:
        mask = (mask > 0).astype(np.uint8) * 255
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    bg = np.full(frame.shape, bg_color, dtype=np.uint8)
    fg = cv2.bitwise_and(frame, frame, mask=mask)
    bg = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask))
    result = cv2.add(fg, bg)
    result = clean_hair_area(frame, result, mask, bg_color)
    return result

def clean_hair_area(original, processed, mask, bg_color):
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    hair_edge_mask = cv2.subtract(dilated_mask, mask)
    bg_sample = cv2.bitwise_and(original, original, mask=cv2.bitwise_not(dilated_mask))
    bg_average = cv2.mean(bg_sample)[:3]
    color_distances = np.sqrt(np.sum((original.astype(np.float32) - bg_average) ** 2, axis=2))
    color_distances = (color_distances - color_distances.min()) / (color_distances.max() - color_distances.min())
    alpha = (1 - color_distances) * (hair_edge_mask / 255.0)
    alpha = np.clip(alpha, 0, 1)
    for c in range(3):
        processed[:, :, c] = processed[:, :, c] * (1 - alpha) + bg_color[c] * alpha
    return processed

with torch.inference_mode():
    checkpoint = 'sam2_hiera_large.pt'
    model_cfg = 'sam2_hiera_l.yaml'
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    body_detector = fasterrcnn_resnet50_fpn(pretrained=True)
    body_detector.eval()
    body_detector.to("cuda")

def download_file(url, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    file_name = url.split('/')[-1]
    file_path = os.path.join(save_dir, file_name)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_video = values['input_video']
    input_video = download_file(url=input_video, save_dir='/content')
    bg_color = values['bg_color']

    bg_color = tuple(int(bg_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))[::-1]
    frames_dir = "/content/frames"
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    ffmpeg_cmd = ["ffmpeg", "-i", str(input_video), "-q:v", "2", "-start_number", "0",f"{frames_dir}/%05d.jpg"]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
    frame_names = [p for p in os.listdir(frames_dir) if p.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG'))]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(video_path=frames_dir)
    first_frame_path = os.path.join(frames_dir, frame_names[0])
    first_frame = cv2.imread(first_frame_path)
    keypoints = detect_body_keypoints(first_frame)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(inference_state=inference_state, frame_idx=0, obj_id=1, points=keypoints, labels=np.ones(len(keypoints), dtype=np.int32))
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: out_mask_logits[i].cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    output_frames_dir = '/content/output_frames'
    os.makedirs(output_frames_dir, exist_ok=True)
    frame_count = 0
    for out_frame_idx in range(len(frame_names)):
        frame_path = os.path.join(frames_dir, frame_names[out_frame_idx])
        frame = cv2.imread(frame_path)
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            frame_with_bg_removed = remove_background(frame, out_mask, bg_color)
        output_frame_path = os.path.join(output_frames_dir, f"{out_frame_idx:05d}.jpg")
        cv2.imwrite(output_frame_path, frame_with_bg_removed)
        frame_count += 1
    output_video_path = '/content/sam2_rm_bg_tost.mp4'
    final_video_cmd = ["ffmpeg", "-y", "-framerate", "30", "-i", f"{output_frames_dir}/%05d.jpg", "-c:v", "libx264", "-pix_fmt", "yuv420p", output_video_path]
    result = subprocess.run(final_video_cmd, capture_output=True, text=True, check=True)

    result = "/content/sam2_rm_bg_tost.mp4"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)
        if os.path.exists(output_frames_dir):
            shutil.rmtree(output_frames_dir)
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)

runpod.serverless.start({"handler": generate})