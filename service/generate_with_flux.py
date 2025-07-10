import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import os
import random
from requests_toolbelt import MultipartEncoder
from PIL import Image
import io
from deep_translator import GoogleTranslator

def open_websocket_connection():
    server_address='127.0.0.1:8188'
    client_id=str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    return ws, server_address, client_id

def queue_prompt(prompt, client_id, server_address):
    p = {"prompt": prompt, "client_id": client_id}
    headers = {'Content-Type': 'application/json'}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data, headers=headers)
    return json.loads(urllib.request.urlopen(req).read())

def get_history(prompt_id, server_address):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())
    
def get_image(filename, subfolder, folder_type, server_address):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def upload_image(input_path, name, server_address, image_type="input", overwrite=False):
    with open(input_path, 'rb') as file:
        multipart_data = MultipartEncoder(
        fields= {
            'image': (name, file, 'image/png'),
            'type': image_type,
            'overwrite': str(overwrite).lower()
        }
        )

        data = multipart_data
        headers = { 'Content-Type': multipart_data.content_type }
        request = urllib.request.Request("http://{}/upload/image".format(server_address), data=data, headers=headers)
        with urllib.request.urlopen(request) as response:
            return response.read()
    
def load_workflow(workflow_path):
    try:
        with open(workflow_path, 'r') as file:
            workflow = json.load(file)
            return json.dumps(workflow)
    except FileNotFoundError:
        print(f"The file {workflow_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"The file {workflow_path} contains invalid JSON.")
        return None

def track_progress(prompt, ws, prompt_id):
    node_ids = list(prompt.keys())
    finished_nodes = []

    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'progress':
                data = message['data']
                current_step = data['value']
                print('In K-Sampler -> Step: ', current_step, ' of: ', data['max'])
            if message['type'] == 'execution_cached':
                data = message['data']
                for itm in data['nodes']:
                    if itm not in finished_nodes:
                        finished_nodes.append(itm)
                        print('Progess: ', len(finished_nodes), '/', len(node_ids), ' Tasks done')
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] not in finished_nodes:
                    finished_nodes.append(data['node'])
                    print('Progess: ', len(finished_nodes), '/', len(node_ids), ' Tasks done')

                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            continue
    return

def get_images(prompt_id, server_address, allow_preview = False):
    output_images = []

    history = get_history(prompt_id, server_address)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        output_data = {}
        if 'images' in node_output:
            for image in node_output['images']:
                if allow_preview and image['type'] == 'temp':
                    preview_data = get_image(image['filename'], image['subfolder'], image['type'], server_address)
                    output_data['image_data'] = preview_data
                if image['type'] == 'output':
                    image_data = get_image(image['filename'], image['subfolder'], image['type'], server_address)
                    output_data['image_data'] = image_data
        output_data['file_name'] = image['filename']
        output_data['type'] = image['type']
        output_images.append(output_data)

    return output_images

def save_image(images, output_path, save_previews):
    for itm in images:
        directory = os.path.join(output_path, 'temp/') if itm['type'] == 'temp' and save_previews else output_path
        os.makedirs(directory, exist_ok=True)
        try:
            image = Image.open(io.BytesIO(itm['image_data']))
            image.save(os.path.join(directory, itm['file_name']))
        except Exception as e:
            print(f"Failed to save image {itm['file_name']}: {e}")  

def generate_image_by_prompt(prompt, output_path, save_previews=False):
    try:
        ws, server_address, client_id = open_websocket_connection()
        print(ws)
        prompt_id = queue_prompt(prompt, client_id, server_address)['prompt_id']
        track_progress(prompt, ws, prompt_id)
        images = get_images(prompt_id, server_address, save_previews)
        save_image(images, output_path, save_previews)
        return images
    finally:
        ws.close()

def prompt_to_image(input_prompt, save_previews=False):
    # Tạo thư mục output nếu chưa tồn tại
    output_dir = '/mnt/data1tb/thangcn/datnv2/images'
    os.makedirs(output_dir, exist_ok=True)
    
    with open('/mnt/data1tb/thangcn/datnv2/workflow/flux_begin.json', 'r') as file:
        workflow = file.read()
    
    prompt = json.loads(workflow)
    id_to_class_type = {id: details['class_type'] for id, details in prompt.items()}
    
    # Thiết lập seed ngẫu nhiên
    random_noise = [key for key, value in id_to_class_type.items() if value == 'RandomNoise'][0]
    prompt.get(random_noise)['inputs']['noise_seed'] = random.randint(10**14, 10**15 - 1)
    
    # Thiết lập prompt văn bản
    text_encoder = [key for key, value in id_to_class_type.items() if value == 'CLIPTextEncode'][0]
    prompt.get(text_encoder)['inputs']['text'] = input_prompt
    
    # Tạo ảnh
    images_data = generate_image_by_prompt(prompt, output_dir, save_previews)
    
    # Trả về danh sách đường dẫn ảnh
    image_paths = []
    for img_data in images_data:
        filename = img_data['file_name']
        img_type = img_data['type']
        
        # Xác định thư mục đích dựa trên loại ảnh
        if img_type == 'temp' and save_previews:
            save_dir = os.path.join(output_dir, 'temp')
        else:
            save_dir = output_dir
        
        # Tạo đường dẫn đầy đủ
        full_path = os.path.join(save_dir, filename)
        image_paths.append(full_path)
    
    return image_paths

# result = GoogleTranslator(source='vi', target='en').translate('đau bụng khi chạy là biểu hiện của bệnh gì')
# prompt_to_image(result, save_previews=True)