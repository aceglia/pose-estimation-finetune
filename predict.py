"""
Exemple d'utilisation du modèle entraîné pour faire des prédictions
"""
import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

import config
from visualize import predict_and_visualize, extract_keypoints_from_heatmaps


def predict_on_image(model_path, image_path, output_path=None):
    """
    Fait une prédiction sur une seule image
    
    Args:
        model_path: Chemin vers le modèle .h5
        image_path: Chemin vers l'image
        output_path: Chemin pour sauvegarder la visualisation (optionnel)
    """
    from tensorflow import keras
    
    print("=" * 60)
    print("🔮 PRÉDICTION SUR UNE IMAGE")
    print("=" * 60)
    
    # Charger le modèle
    print(f"\n📂 Chargement du modèle: {model_path}")
    model = keras.models.load_model(model_path)
    print("✅ Modèle chargé")
    
    # Faire la prédiction et visualiser
    print(f"\n📷 Prédiction sur: {image_path}")
    heatmaps, keypoints = predict_and_visualize(model, image_path, save_path=output_path)
    
    # Afficher les résultats
    print("\n📊 Résultats:")
    print(f"   - Heatmaps shape: {heatmaps.shape}")
    print(f"   - Keypoints détectés: {len(keypoints)}")
    
    return heatmaps, keypoints


def predict_on_folder(model_path, folder_path, output_dir=None, max_images=None):
    """
    Fait des prédictions sur toutes les images d'un dossier
    
    Args:
        model_path: Chemin vers le modèle .h5
        folder_path: Dossier contenant les images
        output_dir: Dossier pour sauvegarder les visualisations
        max_images: Nombre maximum d'images à traiter (None = toutes)
    """
    from tensorflow import keras
    
    print("=" * 60)
    print("🔮 PRÉDICTION SUR UN DOSSIER")
    print("=" * 60)
    
    # Charger le modèle
    print(f"\n📂 Chargement du modèle: {model_path}")
    model = keras.models.load_model(model_path)
    print("✅ Modèle chargé")
    
    # Créer le dossier de sortie
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n💾 Résultats sauvegardés dans: {output_dir}")
    
    # Lister les images
    folder = Path(folder_path)
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    images = [f for f in folder.iterdir() 
              if f.suffix.lower() in image_extensions]
    
    if max_images:
        images = images[:max_images]
    
    print(f"\n📷 {len(images)} images trouvées")
    
    # Prédire sur chaque image
    all_keypoints = []
    
    for i, image_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] Traitement: {image_path.name}")
        
        save_path = None
        if output_dir:
            save_path = os.path.join(output_dir, f"pred_{image_path.stem}.png")
        
        heatmaps, keypoints = predict_and_visualize(
            model, str(image_path), save_path=save_path
        )
        
        all_keypoints.append({
            'image': image_path.name,
            'keypoints': keypoints
        })
    
    # Sauvegarder les résultats dans un fichier
    if output_dir:
        results_file = os.path.join(output_dir, "predictions.txt")
        with open(results_file, 'w') as f:
            f.write("IMAGE,BODYPART,X,Y\n")
            for result in all_keypoints:
                for i, bodypart in enumerate(config.BODYPARTS):
                    x, y = result['keypoints'][i]
                    f.write(f"{result['image']},{bodypart},{x:.4f},{y:.4f}\n")
        
        print(f"\n💾 Résultats sauvegardés: {results_file}")
    
    print("\n✅ Traitement terminé!")
    return all_keypoints


def predict_with_tflite(tflite_path, image_path):
    """
    Fait une prédiction avec le modèle TFLite
    
    Args:
        tflite_path: Chemin vers le modèle .tflite
        image_path: Chemin vers l'image
    """
    import tensorflow as tf
    
    print("=" * 60)
    print("🔮 PRÉDICTION AVEC TFLITE")
    print("=" * 60)
    
    # Charger l'interpréteur
    print(f"\n📂 Chargement du modèle TFLite: {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✅ Modèle chargé")
    print(f"   - Input shape: {input_details[0]['shape']}")
    print(f"   - Output shape: {output_details[0]['shape']}")
    
    # Charger et prétraiter l'image
    print(f"\n📷 Chargement de l'image: {image_path}")
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, config.IMAGE_SIZE)
    img = img.astype(np.float32) / 255.0
    
    # Préparer l'entrée
    input_data = np.expand_dims(img, axis=0).astype(np.float32)
    
    # Si quantization, convertir en uint8
    if input_details[0]['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details[0]['quantization']
        input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)
    
    # Inférence
    print("\n🔄 Inférence en cours...")
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Récupérer la sortie
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Si quantization, déquantizer
    if output_details[0]['dtype'] == np.uint8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
    
    heatmaps = output_data[0]
    
    # Extraire les keypoints
    keypoints = extract_keypoints_from_heatmaps(heatmaps)
    
    print("\n✅ Prédiction terminée!")
    print(f"\n📍 Keypoints détectés:")
    for i, bodypart in enumerate(config.BODYPARTS):
        print(f"   {bodypart}: x={keypoints[i][0]:.3f}, y={keypoints[i][1]:.3f}")
    
    return heatmaps, keypoints


def main():
    parser = argparse.ArgumentParser(description="Prédictions avec le modèle entraîné")
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help="Chemin vers le modèle .h5 ou .tflite"
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help="Chemin vers une image"
    )
    parser.add_argument(
        '--folder',
        type=str,
        default=None,
        help="Chemin vers un dossier d'images"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/predictions',
        help="Dossier de sortie pour les visualisations"
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help="Nombre max d'images à traiter (dossier)"
    )
    parser.add_argument(
        '--tflite',
        action='store_true',
        help="Utiliser le modèle TFLite au lieu de .h5"
    )
    
    args = parser.parse_args()
    
    # Vérifier les arguments
    if not args.model:
        # Chercher le dernier modèle dans tous les dossiers de modèles
        output_dir = Path(config.OUTPUT_DIR)
        models = []
        
        # Parcourir tous les dossiers de modèles
        for model_dir in output_dir.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith('.'):
                models_subdir = model_dir / "models"
                if models_subdir.exists():
                    if args.tflite:
                        models.extend(list(models_subdir.glob("*.tflite")))
                    else:
                        models.extend(list(models_subdir.glob("*_best.h5")))
        
        if models:
            args.model = str(max(models, key=os.path.getctime))
            print(f"💡 Utilisation du modèle: {args.model}")
        else:
            print("❌ Aucun modèle trouvé!")
            print("💡 Spécifiez un modèle avec --model")
            return
    
    if not args.image and not args.folder:
        print("❌ Vous devez spécifier --image ou --folder")
        return
    
    # Faire les prédictions
    try:
        # Déterminer le dossier de sortie basé sur le modèle actuel
        model_path = Path(args.model)
        model_dir = model_path.parent.parent  # Remonter de models/ vers le dossier du modèle
        default_output_dir = model_dir / "predictions"
        
        if args.tflite:
            if args.image:
                predict_with_tflite(args.model, args.image)
            else:
                print("❌ --tflite supporte seulement --image pour le moment")
        else:
            if args.image:
                predict_on_image(args.model, args.image, output_path=str(default_output_dir / "prediction_single.png"))
            elif args.folder:
                predict_on_folder(args.model, args.folder, str(default_output_dir), args.max_images)
    
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        raise


if __name__ == "__main__":
    main()
