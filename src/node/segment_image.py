from src.utils.config import *
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import io
import base64

def extract_fashion_items(image_array, segment_map):
    """Extract individual fashion items as PNG images"""
    fashion_items = []
    
    for label in np.unique(segment_map):
        if label not in FASHION_LABELS:  
            continue
            
        
        mask = (segment_map == label).astype(np.uint8)
        
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            
            item_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            
            item_region = image_array[y:y+h, x:x+w]
            item_mask = mask[y:y+h, x:x+w]
            
            # Set RGB channels
            item_rgba[:, :, :3] = item_region
            # Set alpha channel (transparency)
            item_rgba[:, :, 3] = item_mask * 255
            
            # Convert to PIL Image and then to base64
            item_pil = Image.fromarray(item_rgba, 'RGBA')
            buffer = io.BytesIO()
            item_pil.save(buffer, format='PNG')
            item_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            fashion_items.append({
                'label': int(label),
                'name': FASHION_LABELS[label],
                'name_vi': FASHION_LABELS_VI[label],
                'color': FASHION_COLORS[label],
                'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'item_png': item_b64,
                'size': {'width': int(w), 'height': int(h)}
            })
    
    return fashion_items


def segment_image(segment_model, image):

    image = Image.open(io.BytesIO(image)).convert("RGB")
    image_array = np.array(image)
    
    inputs = segment_model.processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = segment_model.model(**inputs)

    logits = outputs.logits
    
    
    upsampled_logits = F.interpolate(
        logits,
        size=image.size[::-1],  # PIL size is (width, height), need (height, width)
        mode="bilinear",
        align_corners=False,
    )
    
    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    
    fashion_items = extract_fashion_items(image_array, pred_seg)
    
    orig_buffer = io.BytesIO()
    image.save(orig_buffer, format='PNG')
    original_b64 = base64.b64encode(orig_buffer.getvalue()).decode('utf-8')
    
    print(f"Fashion analysis completed: {len(fashion_items)} items detected")
    
    return {
        "success": True,
        "original_image": original_b64,
        "fashion_items": fashion_items,
        "image_size": {"width": image.width, "height": image.height},
        "total_items": len(fashion_items)
    }
    


