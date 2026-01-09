"""
Test du modèle Keras (.h5) sur une vidéo
"""
import cv2
import numpy as np
from tensorflow import keras
import argparse
import os
from pathlib import Path
import config
from preprocessing_utils import decode_and_resize
from validation_utils import from_heatmaps_to_coords, project_keypoints



def predict_frame(model, frame):
    """Fait une prédiction sur une frame"""
    input_data, padding = decode_and_resize(frame)
    heatmaps = model.predict(input_data, verbose=0)[0]
    return heatmaps, padding


def draw_keypoints(frame, heatmap, labels=None, padding=(0, 0)):
    """Dessine les keypoints sur la frame"""
    keypoints = from_heatmaps_to_coords(heatmap, from_prediction=True)
    keypoints = project_keypoints(keypoints, frame.shape[:-1], )

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

        # Ajouter le label et la confiance
        label = f"{labels[i]}: {kp['confidence']:.2f}"
        cv2.putText(frame, label, (kp['x'] + 15, kp['y'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Dessiner les connexions (squelette)
    if len(keypoints) >= 3:
        # Hanche -> Genoux
        cv2.line(frame, (keypoints[0]['x'], keypoints[0]['y']), 
                (keypoints[1]['x'], keypoints[1]['y']), (255, 255, 0), 2)
        # Genoux -> Cheville
        cv2.line(frame, (keypoints[1]['x'], keypoints[1]['y']), 
                (keypoints[2]['x'], keypoints[2]['y']), (255, 255, 0), 2)

    return frame


def process_video(video_path, model_path, output_path=None):
    """Traite une vidéo complète"""

    # Charger le modèle
    model = keras.models.load_model(model_path)

    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")

    # Propriétés de la vidéo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Prédiction
        heatmaps, padding = predict_frame(model, frame)

        # Dessiner keypoints
        annotated_frame = frame.copy()
        draw_keypoints(annotated_frame, heatmaps, None, padding)

        # Écrire la frame
        out.write(annotated_frame)

        frame_count += 1

        # Afficher le progrès
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Progrès: {progress:.1f}% ({frame_count}/{total_frames})")

    cap.release()
    out.release()

    print(f"\nTraitement terminé: {frame_count} frames traitées")

    return output_path

def parse_args(video_path=None, output=None, model=None):
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Test du modèle Keras sur une vidéo")
    parser.add_argument('--video', default=video_path, help='Chemin vers la vidéo à analyser')
    parser.add_argument('--model', default=model,
                       help='Chemin vers le modèle Keras (.keras)')
    parser.add_argument('--output', default=output, help='Chemin de sortie pour la vidéo annotée')

    args = parser.parse_args()
    return args

def main(args):
    # Vérifier que la vidéo existe
    if not os.path.exists(args.video):
        print(f"Vidéo non trouvée: {args.video}")
        return
    
    # Output par défaut
    if not args.output:
        video_name = Path(args.video).stem
        # Utiliser le dossier videos du modèle actuel
        model_path = Path(args.model)
        model_dir = model_path.parent.parent  # Remonter de models/ vers le dossier du modèle
        videos_dir = model_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        args.output = str(videos_dir / f"{video_name}_keras_annotated.mp4")

    # Traiter la vidéo
    try:
        process_video(args.video, args.model, args.output)

    except Exception as e:
        print(f"Erreur lors du traitement: {e}")


if __name__ == "__main__":
    video_path = "/mnt/c/Users/Usager/Documents/Amedeo/PFE/PFE/VideoBrute/Video_Trie_Apres100/115G.mp4"
    model = "/mnt/c/Users/Usager/Documents/Amedeo/pose-estimation-finetune/output/MNv3S_20260108_103408/models/pose_model_backbone_final.keras"
    output = f"/mnt/c/Users/Usager/Documents/Amedeo/pose-estimation-finetune/output/MNv3S_20260108_103408/videos/115G_keras_annotated.mp4"
    args = parse_args(video_path, output, model)
    main(args)
