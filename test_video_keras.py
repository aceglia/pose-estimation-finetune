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


def load_keras_model(model_path):
    """Charge le modèle Keras"""
    print(f"🔄 Chargement du modèle Keras...")
    model = keras.models.load_model(model_path)
    print("✅ Modèle chargé")
    return model


def preprocess_frame(frame, input_size=(192, 192)):
    """Prétraite une frame pour le modèle"""
    frame_resized = cv2.resize(frame, input_size)
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    frame_batch = np.expand_dims(frame_normalized, axis=0)
    return frame_batch


def predict_frame(model, frame):
    """Fait une prédiction sur une frame"""
    input_data = preprocess_frame(frame)
    heatmaps = model.predict(input_data, verbose=0)[0]
    return heatmaps


def extract_keypoints_from_heatmaps(heatmaps, frame_shape):
    """Extrait les coordonnées des keypoints depuis les heatmaps"""
    h, w = frame_shape[:2]
    keypoints = []

    for i in range(heatmaps.shape[-1]):
        heatmap = heatmaps[:, :, i]
        max_pos = np.unravel_index(heatmap.argmax(), heatmap.shape)
        y = int(max_pos[0] * h / heatmap.shape[0])
        x = int(max_pos[1] * w / heatmap.shape[1])
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
    print(f"💡 Utilisation du modèle: {model_path}")
    print(f"📹 Vidéo: {video_path}")

    # Charger le modèle
    model = load_keras_model(model_path)

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

    # Préparer la sortie
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"output/{video_name}_keras_annotated.mp4"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\n💾 Sortie: {output_path}")

    frame_count = 0
    print("\n🔄 Traitement des frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Prédiction
        heatmaps = predict_frame(model, frame)

        # Extraire keypoints
        keypoints = extract_keypoints_from_heatmaps(heatmaps, (height, width))

        # Dessiner keypoints
        annotated_frame = frame.copy()
        draw_keypoints(annotated_frame, keypoints)

        # Écrire la frame
        out.write(annotated_frame)

        frame_count += 1

        # Afficher le progrès
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Progrès: {progress:.1f}% ({frame_count}/{total_frames})")

    cap.release()
    out.release()

    print(f"\n✅ Traitement terminé: {frame_count} frames traitées")
    print(f"🎉 Vidéo annotée sauvegardée: {output_path}")

    return output_path


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Test du modèle Keras sur une vidéo")
    parser.add_argument('--video', required=True, help='Chemin vers la vidéo à analyser')
    parser.add_argument('--model', default='output/models/pose_model_20251105_115946_best.h5',
                       help='Chemin vers le modèle Keras (.h5)')
    parser.add_argument('--output', help='Chemin de sortie pour la vidéo annotée')

    args = parser.parse_args()

    # Vérifier que la vidéo existe
    if not os.path.exists(args.video):
        print(f"❌ Vidéo non trouvée: {args.video}")
        return

    # Vérifier que le modèle existe
    if not os.path.exists(args.model):
        print(f"❌ Modèle non trouvé: {args.model}")
        return

    # Traiter la vidéo
    try:
        output_path = process_video(args.video, args.model, args.output)
        print("\n💡 Touches:")
        print("   - 'q': Quitter")
        print("   - 'espace': Pause/Resume")
    except Exception as e:
        print(f"❌ Erreur lors du traitement: {e}")


if __name__ == "__main__":
    main()
