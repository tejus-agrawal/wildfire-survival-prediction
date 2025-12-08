import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import PolynomialFeatures

# --- Data Cleaning ---

def map_damage(val):
    """
    Maps damage string to binary target: 1 (Burned/Major), 0 (Survived/Minor).
    """
    val = str(val).lower()
    # BURNED (1)
    if 'destroyed' in val or 'major' in val:
        return 1
    # SURVIVED (0)
    # Includes 'Minor', 'Affected', 'No Damage'
    elif 'minor' in val or 'affected' in val or 'no damage' in val:
        return 0
    else:
        return np.nan # Drop ambiguous cases

# --- SAM Extraction / Image Processing ---

def get_house_mask(predictor, image_rgb, center_point=np.array([[320, 320]]), point_label=np.array([1])):
    """Uses SAM to find the central structure."""
    # Predictor image must be set beforehand
    masks, scores, _ = predictor.predict(
        point_coords=center_point,
        point_labels=point_label,
        multimask_output=True
    )
    best_idx = np.argmax(scores)
    house_mask = masks[best_idx].astype(np.uint8)

    # Sanity Check: Reject green houses (Tree canopy errors)
    center_px = image_rgb[center_point[0,1], center_point[0,0]] # y, x
    # If Green is dominant over Red and Blue
    if center_px[1] > center_px[0] + 10 and center_px[1] > center_px[2] + 10:
        return np.zeros_like(house_mask)
    
    return house_mask

def get_vegetation_layers(image_rgb, house_mask):
    """Vectorized Color/Texture analysis to separate Trees vs Grass."""
    # 1. ExG (Excess Green) Index
    r, g, b = image_rgb[:,:,0].astype(float), image_rgb[:,:,1].astype(float), image_rgb[:,:,2].astype(float)
    exg = 2*g - r - b
    
    green_mask = (exg > 10).astype(np.uint8)
    green_mask[house_mask == 1] = 0 
    
    if np.sum(green_mask) == 0:
        return np.zeros_like(green_mask), np.zeros_like(green_mask)

    # 2. Texture Analysis (Variance) to separate Trees vs Grass
    g_channel = image_rgb[:,:,1]
    mean = cv2.boxFilter(g_channel, cv2.CV_32F, (3, 3))
    sq_mean = cv2.boxFilter(g_channel**2, cv2.CV_32F, (3, 3))
    variance = sq_mean - (mean**2)
    
    hsv_v = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)[:,:,2]
    
    # Tree = Green AND (Rough Texture OR Dark Shadows)
    is_rough = variance > 200
    is_dark = hsv_v < 100
    
    tree_mask = np.zeros_like(green_mask)
    tree_mask[(green_mask == 1) & (is_rough | is_dark)] = 1
    
    # Grass = Green AND NOT Tree
    grass_mask = np.zeros_like(green_mask)
    grass_mask[(green_mask == 1) & (tree_mask == 0)] = 1
    
    # 3. Cleanup
    kernel = np.ones((3,3), np.uint8)
    tree_mask = cv2.morphologyEx(tree_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    tree_mask = cv2.morphologyEx(tree_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return tree_mask, grass_mask

def compute_metrics(house_mask, tree_mask, grass_mask, pixels_to_meters=0.07):
    """Calculates tabular features from masks."""
    feats = {}
    
    # Areas
    feats['structure_area_m2'] = np.sum(house_mask) * (pixels_to_meters**2)
    feats['tree_area_m2'] = np.sum(tree_mask) * (pixels_to_meters**2)
    feats['grass_area_m2'] = np.sum(grass_mask) * (pixels_to_meters**2)
    
    # Counts
    nb_trees, _, _, _ = cv2.connectedComponentsWithStats(tree_mask, connectivity=8)
    feats['tree_count'] = max(0, nb_trees - 1)
    
    # Defensible Space
    if np.sum(house_mask) > 0 and np.sum(tree_mask) > 0:
        dist_map = cv2.distanceTransform(1-house_mask, cv2.DIST_L2, 3)
        min_dist_px = np.min(dist_map[tree_mask == 1])
        feats['defensible_space_m'] = min_dist_px * pixels_to_meters
    else:
        feats['defensible_space_m'] = 100.0
        
    # Compactness
    if np.sum(house_mask) > 0:
        contours, _ = cv2.findContours(house_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perim = cv2.arcLength(contours[0], True)
            area = cv2.contourArea(contours[0])
            feats['compactness'] = (4 * 3.14159 * area) / (perim**2 + 1e-6)
        else: feats['compactness'] = 0
    else: feats['compactness'] = 0
    
    return feats

def create_stack(house_mask, tree_mask, output_size=(224, 224)):
    """Creates 3-Channel Tensor for CNN."""
    if np.sum(house_mask) > 0:
        dist_map = cv2.distanceTransform(1-house_mask, cv2.DIST_L2, 3)
        risk_channel = np.clip(255 - (dist_map * 0.6), 0, 255).astype(np.uint8)
        risk_channel = cv2.bitwise_and(risk_channel, risk_channel, mask=tree_mask)
    else:
        risk_channel = np.zeros_like(house_mask)
        
    stack = np.dstack([house_mask * 255, tree_mask * 255, risk_channel])
    stack_resized = cv2.resize(stack, output_size, interpolation=cv2.INTER_NEAREST)
    return stack_resized

# --- MLP Feature Engineering ---

def load_and_engineer_features(filepath):
    df = pd.read_csv(filepath)
    
    required_cols = ['structure_area_m2', 'tree_area_m2', 'grass_area_m2', 'defensible_space_m']
    if not all(col in df.columns for col in required_cols):
        print("⚠️ Warning: Physics columns missing. Skipping physics engineering.")
        # Fallback
        feature_cols = [c for c in df.columns if c not in ['id', 'address', 'damage_str', 'structure_type', 'target', 'lat', 'lon', 'filename']]
        return df, feature_cols.values, feature_cols

    # 1. Total Lot Area Proxy
    df['estimated_lot_area'] = df['structure_area_m2'] + df['tree_area_m2'] + df['grass_area_m2'] + 1.0
    
    # 2. Densities
    df['structure_density'] = df['structure_area_m2'] / df['estimated_lot_area']
    df['fuel_density'] = (df['tree_area_m2'] * 1.5) / df['estimated_lot_area']
    
    # 3. Danger Index
    df['danger_index'] = df['tree_area_m2'] / (df['defensible_space_m'] + 0.1)
    
    # 4. Log Transforms
    skewed = ['structure_area_m2', 'tree_area_m2', 'grass_area_m2', 'defensible_space_m', 'danger_index']
    for col in skewed:
        df[col] = np.log1p(df[col])
        
    base_features = [
        'structure_area_m2', 'tree_area_m2', 'grass_area_m2', 
        'defensible_space_m', 'structure_density', 'fuel_density', 'danger_index'
    ]
    
    # --- B. Interaction Terms ---
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    interactions = poly.fit_transform(df[base_features])
    
    target_feature_names = [f"poly_{i}" for i in range(interactions.shape[1])]
    # df_poly = pd.DataFrame(interactions, columns=target_feature_names) # Optional if we returned df
    
    X = interactions
    y = df['target'].values
    
    return X, y, target_feature_names

