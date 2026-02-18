import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
import datetime

# Initialize Firebase App
cred_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'serviceAccountKey.json')
if os.path.exists(cred_path):
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'civicissue-aae6d.firebasestorage.app' 
    })
    db = firestore.client()
    bucket = storage.bucket()
else:
    print(f"Warning: {cred_path} not found. Firebase features will not work.")
    db = None
    bucket = None

def get_camera_config(camera_id):
    """
    Fetch camera configuration and ROIs from Firestore.
    """
    if not db:
        return None
    
    try:
        # Get Camera Document
        doc_ref = db.collection('cameras').document(camera_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            print(f"Camera {camera_id} not found.")
            return None
            
        camera_data = doc.to_dict()
        camera_data['id'] = camera_id
        
        # Get Regions Subcollection
        regions_ref = doc_ref.collection('regions')
        regions_docs = regions_ref.stream()
        
        rois = []
        for region_doc in regions_docs:
            region_data = region_doc.to_dict()
            
            # Format points: [{'x': 0.1, 'y': 0.2}, ...] -> [[0.1, 0.2], ...]
            # Handle both list of dicts or map of dicts if indices are keys
            points_raw = region_data.get('points', [])
            points = []
            
            if isinstance(points_raw, list):
                # Ensure sorted by index if needed, but usually list preserves order
                for p in points_raw:
                    if 'x' in p and 'y' in p:
                        points.append([float(p['x']), float(p['y'])])
            elif isinstance(points_raw, dict):
                # If stored as map with index keys "0", "1", ...
                sorted_keys = sorted(points_raw.keys(), key=lambda k: int(k) if str(k).isdigit() else k)
                for k in sorted_keys:
                    p = points_raw[k]
                    if 'x' in p and 'y' in p:
                        points.append([float(p['x']), float(p['y'])])
            
            rois.append({
                'id': region_doc.id,
                'type': region_data.get('type', 'unknown'),
                'label': region_data.get('label', ''),
                'points': points
            })
            
        camera_data['rois'] = rois
        return camera_data

    except Exception as e:
        print(f"Error fetching camera config: {e}")
        return None

def create_incident(incident_data):
    """
    Create an incident record in Firestore.
    incident_data: dict containing 'type', 'confidence', 'camera_id', 'timestamp', 'bbox', 'roi_id'
    """
    if not db:
        print("Firebase DB not initialized. skipping incident creation.")
        return

    try:
        # Use complain_id as document ID provided
        complain_id = incident_data.get('complain_id')
        
        if complain_id:
            db.collection('issues').document(complain_id).set(incident_data)
            print(f"Incident created: {complain_id} at {datetime.datetime.now()}")
        else:
            # Fallback for testing or missing ID
            incident_data['created_at'] = firestore.SERVER_TIMESTAMP
            db.collection('issues').add(incident_data)
            print(f"Incident created (auto-id) at {datetime.datetime.now()}")

    except Exception as e:
        print(f"Error creating incident: {e}")

def upload_incident_image(image_path, remote_path):
    """
    Upload an image to Firebase Storage.
    """
    if not bucket:
        return None

    try:
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(image_path)
        blob.make_public()
        return blob.public_url
    except Exception as e:
        print(f"Error uploading image: {e}")
        return None
