from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image
import io
import re
import colorsys
import webcolors

app = Flask(__name__)
CORS(app) # Basic CORS

# --- CONFIGURATION & DICTIONARIES ---
CSS3_HEX_TO_NAMES = {
    '#f0f8ff': 'aliceblue', '#faebd7': 'antiquewhite', '#00ffff': 'aqua', '#7fffd4': 'aquamarine', '#f0ffff': 'azure',
    '#f5f5dc': 'beige', '#ffe4c4': 'bisque', '#000000': 'black', '#ffebcd': 'blanchedalmond', '#0000ff': 'blue',
    '#8a2be2': 'blueviolet', '#a52a2a': 'brown', '#deb887': 'burlywood', '#5f9ea0': 'cadetblue', '#7fff00': 'chartreuse',
    '#d2691e': 'chocolate', '#ff7f50': 'coral', '#6495ed': 'cornflowerblue', '#fff8dc': 'cornsilk', '#dc143c': 'crimson',
    '#00ffff': 'cyan', '#00008b': 'darkblue', '#008b8b': 'darkcyan', '#b8860b': 'darkgoldenrod', '#a9a9a9': 'darkgray',
    '#a9a9a9': 'darkgrey', '#006400': 'darkgreen', '#bdb76b': 'darkkhaki', '#8b008b': 'darkmagenta', '#556b2f': 'darkolivegreen',
    '#ff8c00': 'darkorange', '#9932cc': 'darkorchid', '#8b0000': 'darkred', '#e9967a': 'darksalmon', '#8fbc8f': 'darkseagreen',
    '#483d8b': 'darkslateblue', '#2f4f4f': 'darkslategray', '#2f4f4f': 'darkslategrey', '#00ced1': 'darkturquoise',
    '#9400d3': 'darkviolet', '#ff1493': 'deeppink', '#00bfff': 'deepskyblue', '#696969': 'dimgray', '#696969': 'dimgrey',
    '#1e90ff': 'dodgerblue', '#b22222': 'firebrick', '#fffaf0': 'floralwhite', '#228b22': 'forestgreen', '#ff00ff': 'fuchsia',
    '#dcdcdc': 'gainsboro', '#f8f8ff': 'ghostwhite', '#ffd700': 'gold', '#daa520': 'goldenrod', '#808000': 'gray',
    '#808080': 'grey', '#008000': 'green', '#adff2f': 'greenyellow', '#f0fff0': 'honeydew', '#ff69b4': 'hotpink',
    '#cd5c5c': 'indianred', '#4b0082': 'indigo', '#fffff0': 'ivory', '#f0e68c': 'khaki', '#e6e6fa': 'lavender',
    '#fff0f5': 'lavenderblush', '#7cfc00': 'lawngreen', '#fffacd': 'lemonchiffon', '#add8e6': 'lightblue', '#f08080': 'lightcoral',
    '#e0ffff': 'lightcyan', '#fafad2': 'lightgoldenrodyellow', '#d3d3d3': 'lightgray', '#d3d3d3': 'lightgrey', '#90ee90': 'lightgreen',
    '#ffb6c1': 'lightpink', '#ffa07a': 'lightsalmon', '#20b2aa': 'lightseagreen', '#87cefa': 'lightskyblue', '#778899': 'lightslategray',
    '#778899': 'lightslategrey', '#b0c4de': 'lightsteelblue', '#ffffe0': 'lightyellow', '#00ff00': 'lime', '#32cd32': 'limegreen',
    '#faf0e6': 'linen', '#ff00ff': 'magenta', '#800000': 'maroon', '#66cdaa': 'mediumaquamarine', '#0000cd': 'mediumblue',
    '#ba55d3': 'mediumorchid', '#9370db': 'mediumpurple', '#3cb371': 'mediumseagreen', '#7b68ee': 'mediumslateblue',
    '#00fa9a': 'mediumspringgreen', '#48d1cc': 'mediumturquoise', '#c71585': 'mediumvioletred', '#191970': 'midnightblue',
    '#f5fffa': 'mintcream', '#ffe4e1': 'mistyrose', '#ffe4b5': 'moccasin', '#ffdead': 'navajowhite', '#000080': 'navy',
    '#fdf5e6': 'oldlace', '#808000': 'olive', '#6b8e23': 'olivedrab', '#ffa500': 'orange', '#ff4500': 'orangered',
    '#da70d6': 'orchid', '#eee8aa': 'palegoldenrod', '#98fb98': 'palegreen', '#afeeee': 'paleturquoise', '#db7093': 'palevioletred',
    '#ffefd5': 'papayawhip', '#ffdab9': 'peachpuff', '#cd853f': 'peru', '#ffc0cb': 'pink', '#dda0dd': 'plum', '#b0e0e6': 'powderblue',
    '#800080': 'purple', '#663399': 'rebeccapurple', '#ff0000': 'red', '#bc8f8f': 'rosybrown', '#4169e1': 'royalblue',
    '#8b4513': 'saddlebrown', '#fa8072': 'salmon', '#f4a460': 'sandybrown', '#2e8b57': 'seagreen', '#fff5ee': 'seashell',
    '#a0522d': 'sienna', '#c0c0c0': 'silver', '#87ceeb': 'skyblue', '#6a5acd': 'slateblue', '#708090': 'slategray',
    '#708090': 'slategrey', '#fffafa': 'snow', '#00ff7f': 'springgreen', '#4682b4': 'steelblue', '#d2b48c': 'tan',
    '#008080': 'teal', '#d8bfd8': 'thistle', '#ff6347': 'tomato', '#40e0d0': 'turquoise', '#ee82ee': 'violet',
    '#f5deb3': 'wheat', '#ffffff': 'white', '#f5f5f5': 'whitesmoke', '#ffff00': 'yellow', '#9acd32': 'yellowgreen'
}

def closest_color(requested_color):
    min_colors = {}
    for hex_value, name in CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_color_name(rgb_tuple):
    try: return webcolors.rgb_to_name(rgb_tuple)
    except ValueError: return closest_color(rgb_tuple)

def suggest_color_count(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.thumbnail((200, 200)) 
        q_img = img.quantize(colors=24, method=2)
        color_counts = q_img.histogram()
        total_pixels = img.width * img.height
        dominant_count = 0
        threshold = total_pixels * 0.005
        for count in color_counts[:24]:
            if count > threshold: dominant_count += 1
        return max(2, dominant_count) 
    except: return 5 

def parse_filename_info(filename):
    name = filename.lower()
    width, height, k_colors = None, None, None
    k_match = re.search(r'[-_ ](\d+)\.(?:png|jpg|jpeg|webp)$', name)
    if k_match: k_colors = int(k_match.group(1))
    dim_match = re.search(r'(\d+(?:\.\d+)?)(?:cm)?\s*[xX]\s*(\d+(?:\.\d+)?)(?:cm)?', name)
    if dim_match:
        width = float(dim_match.group(1))
        height = float(dim_match.group(2))
    else:
        w_match = re.search(r'(?:w|width)[^0-9]*([0-9]+\.?[0-9]*)', name)
        h_match = re.search(r'(?:h|height)[^0-9]*([0-9]+\.?[0-9]*)', name)
        width = float(w_match.group(1)) if w_match else None
        height = float(h_match.group(1)) if h_match else None
    return width, height, k_colors

def process_image_for_analysis(image_bytes, is_dotted=False):
    processing_width = 1200 if is_dotted else 800 
    img = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(img)
    original_h, original_w = img_array.shape[:2]
    scale_factor = processing_width / original_w
    processing_height = int(original_h * scale_factor)
    img_small = cv2.resize(img_array, (processing_width, processing_height), interpolation=cv2.INTER_AREA)
    img_processed = img_small
    if not is_dotted:
        if len(img_small.shape) == 3 and img_small.shape[2] == 4:
            rgb_part = img_small[:, :, :3]
            alpha_part = img_small[:, :, 3]
            rgb_smooth = cv2.bilateralFilter(rgb_part, 5, 50, 50) 
            img_processed = np.dstack((rgb_smooth, alpha_part))
        else:
            img_processed = cv2.bilateralFilter(img_small, 5, 50, 50)
    
    if len(img_processed.shape) == 3 and img_processed.shape[2] == 4:
        rgb_small = img_processed[:, :, :3]
        alpha_small = img_processed[:, :, 3]
        lab_small = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2LAB)
        visible_lab = lab_small[alpha_small > 50]
        visible_rgb = rgb_small[alpha_small > 50]
        visible_ratio = len(visible_lab) / (processing_width * processing_height)
        return visible_lab, visible_rgb, original_h, original_w, visible_ratio
    else:
        lab_small = cv2.cvtColor(img_processed, cv2.COLOR_RGB2LAB)
        visible_lab = lab_small.reshape(-1, 3)
        visible_rgb = img_processed.reshape(-1, 3)
        visible_ratio = 1.0
        return visible_lab, visible_rgb, original_h, original_w, visible_ratio

def consolidate_by_hsv(raw_results):
    if not raw_results: return []
    def get_hsv(rgb):
        r, g, b = [x/255.0 for x in rgb]
        return colorsys.rgb_to_hsv(r, g, b)
    for item in raw_results: item['hsv'] = get_hsv(item['rgb'])
    sorted_items = sorted(raw_results, key=lambda x: x['area'], reverse=True)
    final_groups = []
    while sorted_items:
        parent = sorted_items.pop(0)
        parent_h, parent_s, parent_v = parent['hsv']
        leftovers = []
        for child in sorted_items:
            child_h, child_s, child_v = child['hsv']
            merged = False
            if parent_s < 0.2 and child_s < 0.2:
                if abs(parent_v - child_v) < 0.2: 
                    parent['area'] += child['area']
                    parent['pct'] += child['pct']
                    merged = True
            elif parent_s >= 0.2 and child_s >= 0.2:
                hue_diff = min(abs(parent_h - child_h), 1 - abs(parent_h - child_h))
                if hue_diff < 0.1: 
                    parent['area'] += child['area']
                    parent['pct'] += child['pct']
                    merged = True
            elif parent_v < 0.15 and child_v < 0.3:
                 parent['area'] += child['area']
                 parent['pct'] += child['pct']
                 merged = True
            if not merged: leftovers.append(child)
        final_groups.append(parent)
        sorted_items = leftovers
    return final_groups

def smart_merge_by_dominance(raw_results, target_k):
    clean_results = consolidate_by_hsv(raw_results)
    for item in clean_results:
        h, s, v = item['hsv'] 
        if s > 0.45: item['score'] = item['area'] * 40.0 
        elif s < 0.15 and v > 0.5: item['score'] = item['area'] * 0.8 
        else: item['score'] = item['area'] * 1.0 
    clean_results.sort(key=lambda x: x['score'], reverse=True)
    final_list = clean_results[:target_k]
    leftovers = clean_results[target_k:]
    for loser in leftovers:
        loser_rgb = np.array(loser['rgb'])
        closest_idx = -1
        min_dist = float('inf')
        for i, winner in enumerate(final_list):
            winner_rgb = np.array(winner['rgb'])
            dist = np.linalg.norm(winner_rgb - loser_rgb)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        if closest_idx != -1:
            final_list[closest_idx]['area'] += loser['area']
            final_list[closest_idx]['pct'] += loser['pct']
    final_list.sort(key=lambda x: x['area'], reverse=True)
    return final_list

def analyze_image_logic(file_name, file_data, height_cm, width_cm, k_colors, is_dotted):
    pixels_lab, pixels_rgb, original_h, original_w, visible_ratio = process_image_for_analysis(file_data, is_dotted)
    if len(pixels_lab) == 0: return {"error": "Empty image"}

    search_k = max(40, k_colors * 8)
    clf = KMeans(n_clusters=search_k, n_init=10)
    labels = clf.fit_predict(pixels_lab)
    counts = Counter(labels)
    
    center_colors_rgb = []
    for i in range(search_k):
        cluster_pixels = pixels_rgb[labels == i]
        if len(cluster_pixels) > 0: avg_color = cluster_pixels.mean(axis=0)
        else: avg_color = np.array([0,0,0])
        center_colors_rgb.append(avg_color)

    one_pixel_area_cm2 = (height_cm / original_h) * (width_cm / original_w)
    real_box_area_cm2 = (original_h * original_w) * one_pixel_area_cm2
    total_ink_area_cm2 = real_box_area_cm2 * visible_ratio

    raw_results = []
    for i in range(search_k):
        color_fraction = counts[i] / len(pixels_lab)
        area_cm2 = color_fraction * total_ink_area_cm2
        rgb = tuple([int(c) for c in center_colors_rgb[i]])
        raw_results.append({
            'rgb': rgb, 
            'area': area_cm2, 
            'pct': color_fraction * 100,
            'name': get_color_name(rgb)
        })

    final_results = smart_merge_by_dominance(raw_results, target_k=k_colors)
    
    output = {
        "filename": file_name,
        "total_ink_area": round(total_ink_area_cm2, 2),
        "canvas_area": round(height_cm * width_cm, 2),
        "colors": []
    }
    
    for c in final_results:
        output["colors"].append({
            "name": c['name'],
            "rgb": c['rgb'],
            "area_cm2": round(c['area'], 2),
            "pct": round(c['pct'], 1)
        })
    return output

# --- API ENDPOINT WITH EXPLICIT OPTIONS HANDLING ---
@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    # 1. HANDLE THE PRE-FLIGHT HANDSHAKE
    if request.method == 'OPTIONS':
        # Browser asks: "Can I POST?" -> We say: "Yes, 200 OK"
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response, 200

    # 2. HANDLE THE ACTUAL UPLOAD (POST)
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    filename = file.filename
    file_data = file.read()
    
    auto_w, auto_h, auto_k = parse_filename_info(filename)
    
    manual_w = request.form.get('width', type=float)
    manual_h = request.form.get('height', type=float)
    manual_k = request.form.get('k_colors', type=int)
    
    width = manual_w if manual_w else auto_w
    height = manual_h if manual_h else auto_h
    colors = manual_k if manual_k else (auto_k if auto_k else suggest_color_count(file_data))
    
    if not width or not height:
        return jsonify({
            "status": "partial",
            "message": "Dimensions required",
            "detected_k": colors
        })
        
    is_dotted = "dot" in filename.lower()
    
    result = analyze_image_logic(filename, file_data, height, width, colors, is_dotted)
    return jsonify({"status": "success", "data": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
