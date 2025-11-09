"""
Test du modèle TFLite sur une vidéo
"""
import cv2
import numpy as np
import tensorflow as tf
import argparse
import os
from pathlib import Path
import config


def load_tflite_model(model_path):
    """Charge le modèle TFLite"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def preprocess_frame(frame, input_size=(192, 192)):
    """Prétraite une frame pour le modèle"""
    frame_resized = cv2.resize(frame, input_size)
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    frame_batch = np.expand_dims(frame_normalized, axis=0)
    return frame_batch


def predict_frame(interpreter, input_details, output_details, frame):
    """Fait une prédiction sur une frame"""
    input_data = preprocess_frame(frame)

    if input_details[0]['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details[0]['quantization']
        input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Déquantization si nécessaire
    if output_details[0]['dtype'] == np.uint8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
    
    return output_data[0]


def extract_keypoints_from_heatmaps(heatmaps, frame_shape):
    """Extrait les coordonnées des keypoints depuis les heatmaps"""
    h, w = frame_shape[:2]
    keypoints = []
    
    for i in range(heatmaps.shape[-1]):
        heatmap = heatmaps[:, :, i]
        
        # Trouver le maximum
        max_pos = np.unravel_index(heatmap.argmax(), heatmap.shape)
        
        # Convertir en coordonnées de l'image
        y = int(max_pos[0] * h / heatmap.shape[0])
        x = int(max_pos[1] * w / heatmap.shape[1])
        
        # Confiance (valeur du pic)
        confidence = heatmap[max_pos]
        
        keypoints.append({'x': x, 'y': y, 'confidence': confidence})
    
    return keypoints


def draw_keypoints(frame, keypoints, labels=None):
    """Dessine les keypoints sur la frame"""
    if labels is None:
        labels = config.BODYPARTS
    
    colors = [
        (255, 0, 0),    # Hanche - Rouge
        (0, 255, 0),    # Genoux - Vert
        (0, 0, 255)     # Cheville - Bleu
    ]
    
    for i, kp in enumerate(keypoints):
        color = colors[i % len(colors)]
        
        # Dessiner le point
        cv2.circle(frame, (kp['x'], kp['y']), 8, color, -1)
        cv2.circle(frame, (kp['x'], kp['y']), 10, (255, 255, 255), 2)
        
        # Ajouter le label
        label = f"{labels[i]}: {kp['confidence']:.2f}"
        cv2.putText(frame, label, (kp['x'] + 15, kp['y']), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Dessiner les connexions (squelette)
    if len(keypoints) >= 3:
        # Hanche -> Genoux
        cv2.line(frame, (keypoints[0]['x'], keypoints[0]['y']), 
                (keypoints[1]['x'], keypoints[1]['y']), (255, 255, 0), 2)
        # Genoux -> Cheville
        cv2.line(frame, (keypoints[1]['x'], keypoints[1]['y']), 
                (keypoints[2]['x'], keypoints[2]['y']), (255, 255, 0), 2)
    
    return frame


def process_video(video_path, model_path, output_path=None, display=True):
    """Traite une vidéo avec le modèle TFLite"""
    
    print(f"📹 Vidéo: {video_path}")
    print(f"🤖 Modèle: {model_path}")
    
    # Charger le modèle
    print("\n🔄 Chargement du modèle...")
    interpreter, input_details, output_details = load_tflite_model(model_path)
    print("✅ Modèle chargé")
    
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")
    
    # Propriétés de la vidéo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📊 Propriétés de la vidéo:")
    print(f"   - Résolution: {width}x{height}")
    print(f"   - FPS: {fps}")
    print(f"   - Frames: {total_frames}")
    
    # Préparer l'output
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"\n💾 Sortie: {output_path}")
    
    print("\n🔄 Traitement des frames...")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Prédiction
            heatmaps = predict_frame(interpreter, input_details, output_details, frame)
            
            # Extraire les keypoints
            keypoints = extract_keypoints_from_heatmaps(heatmaps, frame.shape)
            
            # Dessiner sur la frame
            frame_annotated = draw_keypoints(frame.copy(), keypoints)
            
            # Ajouter le numéro de frame
            cv2.putText(frame_annotated, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Sauvegarder
            if out:
                out.write(frame_annotated)
            
            # Afficher
            if display:
                cv2.imshow('Pose Estimation', frame_annotated)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⚠️  Arrêté par l'utilisateur")
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)  # Pause
            
            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   Progrès: {progress:.1f}% ({frame_count}/{total_frames})")
    
    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        if display:
            cv2.destroyAllWindows()
    
    print(f"\n✅ Traitement terminé: {frame_count} frames traitées")
    
    return frame_count


def main():
    parser = argparse.ArgumentParser(description="Test du modèle TFLite sur une vidéo")
    
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help="Chemin vers la vidéo"
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help="Chemin vers le modèle .tflite"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Chemin pour sauvegarder la vidéo annotée"
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help="Ne pas afficher la vidéo en temps réel"
    )
    
    args = parser.parse_args()
    
    # Trouver le modèle si non spécifié
    if not args.model:
        # Chercher dans tous les dossiers de modèles
        output_dir = Path(config.OUTPUT_DIR)
        tflite_models = []
        
        # Parcourir tous les dossiers de modèles
        for model_dir in output_dir.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith('.'):
                models_subdir = model_dir / "models"
                if models_subdir.exists():
                    tflite_models.extend(list(models_subdir.glob("*.tflite")))
        
        if tflite_models:
            # Prendre le plus récent
            args.model = str(max(tflite_models, key=os.path.getctime))
            print(f"💡 Utilisation du modèle le plus récent: {args.model}")
        else:
            print("❌ Aucun modèle .tflite trouvé!")
            print("💡 Entraînez d'abord le modèle avec: python main.py")
            return
    
    # Output par défaut
    if not args.output:
        video_name = Path(args.video).stem
        
        # Déterminer le type de modèle
        model_name = Path(args.model).name
        if 'dynamic' in model_name:
            model_type = 'dynamic'
        elif 'float32' in model_name:
            model_type = 'float32'
        else:
            model_type = 'tflite'
        
        # Utiliser le dossier videos du modèle actuel
        model_path = Path(args.model)
        model_dir = model_path.parent.parent  # Remonter de models/ vers le dossier du modèle
        videos_dir = model_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        args.output = str(videos_dir / f"{video_name}_{model_type}_annotated.mp4")
    
    # Traiter la vidéo
    try:
        process_video(
            args.video, 
            args.model, 
            args.output, 
            display=not args.no_display
        )
        
        print(f"\n🎉 Vidéo annotée sauvegardée: {args.output}")
        print("\n💡 Touches:")
        print("   - 'q': Quitter")
        print("   - 'espace': Pause/Resume")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        raise


if __name__ == "__main__":
    main()
