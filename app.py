import base64
import io
import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from PIL import Image, ImageDraw

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

BAND_PATHS = {
    'B02': "data/sentinel/B02_sentinel-2.tiff",
    'B03': "data/sentinel/B03_sentinel-2.tiff",
    'B04': "data/sentinel/B04_sentinel-2.tiff",
    'B08': "data/sentinel/B08_sentinel-2.tiff",
    'B11': "data/sentinel/B11_sentinel-2.tiff", 
}

@app.route("/")
@app.route("/index.html")
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route("/calculate/<index_name>", methods=["POST"])
def calculate_index(index_name):
    data = request.json
    bbox = data["bbox"]
    geom = [box(*bbox)]

    try:
        with rasterio.open(BAND_PATHS['B04']) as red_src:
            red_clipped, transform = mask(red_src, geom, crop=True)
        with rasterio.open(BAND_PATHS['B08']) as nir_src:
            nir_clipped, _ = mask(nir_src, geom, crop=True)
        
        red = red_clipped[0].astype("float32") / 10000.0
        nir = nir_clipped[0].astype("float32") / 10000.0
        
        h, w = min(red.shape[0], nir.shape[0]), min(red.shape[1], nir.shape[1])
        red = red[:h, :w]
        nir = nir[:h, :w]
        
        bands = {'red': red, 'nir': nir}
        
        for band_code, band_name in [('B02', 'blue'), ('B03', 'green'), ('B11', 'swir')]:
            if os.path.exists(BAND_PATHS[band_code]):
                with rasterio.open(BAND_PATHS[band_code]) as src:
                    clipped, _ = mask(src, geom, crop=True)
                    band_data = clipped[0][:h, :w].astype("float32") / 10000.0
                    bands[band_name] = band_data
            else:
                print(f"{BAND_PATHS[band_code]} не найден")
        
    except Exception as e:
        return jsonify({f"Ошибка данных: {str(e)}"}), 500

    formulas = {
        'ndvi': lambda b: (b['nir'] - b['red']) / (b['nir'] + b['red'] + 1e-6),
        'ndwi': lambda b: (b['nir'] - b.get('swir', b['red'])) / (b['nir'] + b.get('swir', b['red']) + 1e-6),
        'evi':  lambda b: 2.5 * (b['nir'] - b['red']) / (b['nir'] + 6*b['red'] - 7.5*b.get('blue', 0) + 1),
        'savi': lambda b: 1.5 * (b['nir'] - b['red']) / (b['nir'] + b['red'] + 0.5),
        'gndvi': lambda b: (b['nir'] - b.get('green', b['red'])) / (b['nir'] + b.get('green', b['red']) + 1e-6)
    }
    
    index_array = np.clip(formulas[index_name](bands), -1.0, 1.0)
    
    print(f"{index_name.upper()}: min={np.nanmin(index_array):.3f}, max={np.nanmax(index_array):.3f}")
    
    h, w = index_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    rgb[index_array < 0] = [100, 150, 255] 
    rgb[(index_array >= 0) & (index_array < 0.2)] = [200, 200, 200] 
    rgb[(index_array >= 0.2) & (index_array < 0.5)] = [255, 255, 0] 
    rgb[(index_array >= 0.5) & (index_array < 0.8)] = [144, 238, 144] 
    rgb[index_array >= 0.8] = [0, 128, 0] 
    
    img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    colorbar = create_colorbar()
    cbar_buffer = io.BytesIO()
    colorbar.save(cbar_buffer, format="PNG")
    cbar_b64 = base64.b64encode(cbar_buffer.getvalue()).decode()
    
    south, west, north, east = bbox[1], bbox[0], bbox[3], bbox[2]
    
    return jsonify({
        "image": img_b64,
        "colorbar": cbar_b64,
        "bounds": [south, west, north, east],
        "stats": {"min": float(np.nanmin(index_array)), "max": float(np.nanmax(index_array))},
        "index_name": index_name.upper()
    })

def create_colorbar():
    cb = Image.new('RGB', (240, 40), (255, 255, 255))
    draw = ImageDraw.Draw(cb)
    
    segments = [
        (-1.0, 0.0, (100, 150, 255)),  
        (0.0, 0.2, (200, 200, 200)),    
        (0.2, 0.5, (255, 255, 0)),    
        (0.5, 0.8, (144, 238, 144)),  
        (0.8, 1.0, (0, 128, 0))      
    ]
    
    x_start = 10
    for min_val, max_val, color in segments:
        width = int((max_val - min_val) * 60) 
        draw.rectangle([x_start, 10, x_start + width, 30], fill=color)
        x_start += width + 2 
    
    draw.text((12, 32), "-1", fill=(0,0,0), font_size=10)
    draw.text((55, 32), "0", fill=(0,0,0), font_size=10)
    draw.text((95, 32), "0.2", fill=(0,0,0), font_size=10)
    draw.text((140, 32), "0.5", fill=(0,0,0), font_size=10)
    draw.text((190, 32), "1", fill=(0,0,0), font_size=10)
    
    return cb

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
