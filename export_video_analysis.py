"""
Export des résultats d'analyse vidéo en CSV
Compare différents formats de modèles (Keras, TFLite float32, dynamic, int8)
"""
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import config


def load_keras_model(model_path):
    """Charge un modèle Keras"""
    print(f"📦 Chargement modèle Keras: {model_path}")
    model = keras.models.load_model(model_path)
    input_shape = model.input_shape
    input_size = (input_shape[2], input_shape[1]) if input_shape[1] else (192, 192)
    return model, input_size


def load_tflite_model(model_path):
    """Charge un modèle TFLite"""
    print(f"📦 Chargement modèle TFLite: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Extraire la taille d'entrée
    input_shape = input_details[0]['shape']
    input_size = (input_shape[2], input_shape[1])
    
    return interpreter, input_details, output_details, input_size


def preprocess_frame(frame, input_size):
    """Prétraite une frame"""
    frame_resized = cv2.resize(frame, input_size)
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    return np.expand_dims(frame_normalized, axis=0)


def predict_keras(model, frame, input_size):
    """Prédiction avec modèle Keras"""
    input_data = preprocess_frame(frame, input_size)
    heatmaps = model.predict(input_data, verbose=0)[0]
    return heatmaps


def predict_tflite(interpreter, input_details, output_details, frame, input_size):
    """Prédiction avec modèle TFLite"""
    input_data = preprocess_frame(frame, input_size)
    
    # Quantization si nécessaire
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
        max_pos = np.unravel_index(heatmap.argmax(), heatmap.shape)
        
        # Coordonnées dans l'image originale
        y = int(max_pos[0] * h / heatmap.shape[0])
        x = int(max_pos[1] * w / heatmap.shape[1])
        confidence = float(heatmap[max_pos])
        
        keypoints.append({'x': x, 'y': y, 'confidence': confidence})
    
    return keypoints


def analyze_video_with_model(video_path, model_info, model_type):
    """Analyse une vidéo avec un modèle donné"""
    results = []
    
    # Charger le modèle selon le type
    if model_type == 'keras':
        model, input_size = load_keras_model(model_info)
    else:
        interpreter, input_details, output_details, input_size = load_tflite_model(model_info)
    
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"🎬 Analyse: {total_frames} frames ({frame_width}x{frame_height})")
    
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc=f"  {model_type}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Prédiction
        if model_type == 'keras':
            heatmaps = predict_keras(model, frame, input_size)
        else:
            heatmaps = predict_tflite(interpreter, input_details, output_details, frame, input_size)
        
        # Extraire les keypoints
        keypoints = extract_keypoints_from_heatmaps(heatmaps, frame.shape)
        
        # Enregistrer les résultats
        for kp_idx, kp in enumerate(keypoints):
            results.append({
                'frame': frame_idx,
                'model_type': model_type,
                'keypoint': config.BODYPARTS[kp_idx],
                'x': kp['x'],
                'y': kp['y'],
                'confidence': kp['confidence']
            })
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    return results


def compare_models(video_path, model_dir, output_csv):
    """Compare différents formats de modèles"""
    print("\n" + "=" * 60)
    print("📊 EXPORT ANALYSE VIDÉO - COMPARAISON MODÈLES")
    print("=" * 60)
    
    model_dir = Path(model_dir)
    models_dir = model_dir / "models"
    
    # Liste des modèles à tester
    models_to_test = []
    
    # Keras
    keras_model = models_dir / "pose_model_best.h5"
    if keras_model.exists():
        models_to_test.append(('keras', str(keras_model)))
    
    # TFLite float32
    float32_model = models_dir / "pose_model_float32.tflite"
    if float32_model.exists():
        models_to_test.append(('float32', str(float32_model)))
    
    # TFLite dynamic
    dynamic_model = models_dir / "pose_model_dynamic.tflite"
    if dynamic_model.exists():
        models_to_test.append(('dynamic', str(dynamic_model)))
    
    # TFLite int8 (si existe)
    int8_model = models_dir / "pose_model_int8.tflite"
    if int8_model.exists():
        models_to_test.append(('int8', str(int8_model)))
    
    if not models_to_test:
        print("❌ Aucun modèle trouvé!")
        return
    
    print(f"\n📹 Vidéo: {video_path}")
    print(f"🔍 Modèles à tester: {len(models_to_test)}")
    for model_type, _ in models_to_test:
        print(f"   - {model_type}")
    
    # Analyser avec chaque modèle
    all_results = []
    
    for model_type, model_path in models_to_test:
        try:
            results = analyze_video_with_model(video_path, model_path, model_type)
            all_results.extend(results)
        except Exception as e:
            print(f"⚠️  Erreur avec {model_type}: {e}")
            continue
    
    # Créer le DataFrame
    df = pd.DataFrame(all_results)
    
    # Sauvegarder en CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Résultats exportés: {output_path}")
    print(f"📊 Total de lignes: {len(df)}")
    print(f"\n📈 Résumé par modèle:")
    print(df.groupby('model_type')['frame'].count())
    
    # Créer aussi un CSV pivot pour faciliter la comparaison
    pivot_path = output_path.parent / f"{output_path.stem}_pivot.csv"
    
    # Créer une version pivot avec toutes les coordonnées
    pivot_data = []
    for frame in df['frame'].unique():
        frame_data = {'frame': frame}
        for model_type in df['model_type'].unique():
            for keypoint in config.BODYPARTS:
                row = df[(df['frame'] == frame) & 
                        (df['model_type'] == model_type) & 
                        (df['keypoint'] == keypoint)]
                if not row.empty:
                    frame_data[f'{model_type}_{keypoint}_x'] = row['x'].values[0]
                    frame_data[f'{model_type}_{keypoint}_y'] = row['y'].values[0]
                    frame_data[f'{model_type}_{keypoint}_conf'] = row['confidence'].values[0]
        pivot_data.append(frame_data)
    
    df_pivot = pd.DataFrame(pivot_data)
    df_pivot.to_csv(pivot_path, index=False)
    
    print(f"📊 Format pivot exporté: {pivot_path}")
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Export l'analyse vidéo en CSV pour différents modèles"
    )
    
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help="Chemin vers la vidéo à analyser"
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help="Dossier du modèle (ex: output/MNv2_20251113_123456)"
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Chemin du fichier CSV de sortie (défaut: dans le dossier du modèle)"
    )
    
    args = parser.parse_args()
    
    # Output par défaut
    if args.output is None:
        video_name = Path(args.video).stem
        model_dir = Path(args.model_dir)
        args.output = str(model_dir / f"{video_name}_analysis.csv")
    
    # Lancer l'analyse
    try:
        compare_models(args.video, args.model_dir, args.output)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        raise


if __name__ == "__main__":
    main()
