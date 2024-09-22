from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import cv2
import requests
import json
import shutil
import firebase_admin
from firebase_admin import credentials, storage
from roboflow_unwrap import unwrap 
from cerebras.cloud.sdk import Cerebras

app = Flask(__name__)

CORS(app, resources={r'/*': {'origins': '*'}})

cred = credentials.Certificate("./firebaseadminkey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'snapill.appspot.com'
})

client = Cerebras(
    api_key='removed',
)

UPLOAD_FOLDER = './uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/") 
def index(): 
    return "Homepage"

@app.route('/vanguard', methods=['POST'])
def pill_vanguard():
    data = request.json
    message = data.get('message', '')

    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    try:
        response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
            model="llama3.1-8b",
        )
        res = response.choices[0].message.content
        return jsonify({"response": res}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat_with_cerebras():
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    try:
        response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
            model="llama3.1-8b",
        )
        res = response.choices[0].message.content
        return jsonify({"response": res}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/process-video', methods=['POST'])
def process_video_from_firebase():
    data = request.get_json()
    video_url = data.get('videoUrl')
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400

    video_filename = os.path.join(UPLOAD_FOLDER, 'downloaded_video.mp4')
    
    try:
        download_video_from_firebase(video_url, video_filename)
    except Exception as e:
        return jsonify({"error": f"Failed to download video: {str(e)}"}), 500

    frames = segment_video(video_filename)
    frame_urls = []
    for i, frame in enumerate(frames):
        frame_path = f"./frame_{i}.jpg"
        cv2.imwrite(frame_path, frame)

        frame_url = upload_frame_to_firebase(frame_path, f"frames/frame_{i}.jpg")
        frame_urls.append(frame_url)

        # # Call the unwrap function from roboflow_unwrap.py
        # print("Unwrapping: ", i)
        # _,unwrapped_frame = unwrap(frame_url, frame_path)  # The image_url is now the Firebase public URL
        # unwrapped_frame_path = f"./unwrapped_frame_{i}.jpg"
        # cv2.imwrite(unwrapped_frame_path,unwrapped_frame)
        # upload_frame_to_firebase(unwrapped_frame_path, f"unwrapped_frames/unwrapped_frame_{i}.jpg")

    merged_image_path = f"./merged_image.jpg"
    merged_image = merge_images_vertically(frames, merged_image_path)
    merged_url = upload_frame_to_firebase(merged_image_path, f"merged_frames/merged_image.jpg")
    res = call_roboflow_workflow(merged_url)
    print(res["outputs"][0]["open_ai"]["output"])
    res = res["outputs"][0]["open_ai"]["output"]
    json_string = res.replace("```json\n", "").replace("\n```", "")
    extracted_data = json.loads(json_string)

    
    return jsonify({
            "message": "Video processed successfully",
            "medication_data": extracted_data  
        }), 200



def merge_images_vertically(images, output_path):
    widths = [img.shape[1] for img in images]
    max_width = max(widths)
    resized_images = [cv2.resize(img, (max_width, img.shape[0])) if img.shape[1] != max_width else img for img in images]
    merged_image = np.vstack(resized_images)
    cv2.imwrite(output_path, merged_image)
    print(f"Merged image saved at {output_path}")
    return merged_image

def download_video_from_firebase(url, destination):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
    else:
        raise Exception(f"Failed to download video. Status code: {response.status_code}")

def segment_video(video_path, num_frames=6):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = total_frames // num_frames

    selected_frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if ret:
            selected_frames.append(frame)
    
    cap.release()
    return selected_frames

def upload_frame_to_firebase(local_file_path, destination_blob_name):
    bucket = storage.bucket()
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path, content_type='image/jpeg')
    blob.make_public() 
    print("Uploaded! ", blob.public_url)
    return blob.public_url 

def call_roboflow_workflow(image_url):
    workflow_url = "https://detect.roboflow.com/infer/workflows/sriram-kalki/custom-workflow-2"
    headers = {"Content-Type": "application/json"}
    data = {
        "api_key": "qTssMHObChVIcD3e2Yw8",
        "inputs": {
            "image": {"type": "url", "value": image_url}
        }
    }

    response = requests.post(workflow_url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)