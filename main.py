"""
Script principal pour le pipeline de fine-tuning
"""
import os
prefix = "/mnt/c" if os.name == "posix" else "C:"
if not os.name == "posix":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.add_dll_directory(os.path.join(prefix, "Users", "neuromolity-lab", "miniconda3", "envs", "pose_est", "Library", "bin"))

import argparse
import numpy as np
from datetime import datetime
import config
from data_preprocessing import prepare_data
from model import create_model
from train import train_model, save_final_model, evaluate_model, plot_training_history
from export_tflite import export_model, test_tflite_model



def main(args):
    """Pipeline complet de fine-tuning"""
    print("\n" + "=" * 60)
    print("🎯 PIPELINE DE FINE-TUNING - POSE ESTIMATION")
    print("=" * 60)

    # Configurer le backbone si spécifié en argument
    if args.backbone:
        config.BACKBONE = args.backbone
        # Adapter la taille d'image selon le backbone
        if args.backbone in config.BACKBONE_INPUT_SIZES:
            recommended_size = config.BACKBONE_INPUT_SIZES[args.backbone]
            config.IMAGE_SIZE = recommended_size
            config.INPUT_SHAPE = (*recommended_size, 3)
    print(f"\nConfiguration:")
    print(f"   - Backbone: {config.BACKBONE}")
    print(f"   - Input size: {config.INPUT_SHAPE[0]}x{config.INPUT_SHAPE[1]}")


    if not args.skip_training:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_folder_name = config.get_model_folder_name(config.BACKBONE, timestamp)
        model_dir, models_dir, logs_dir, videos_dir = config.setup_model_directories(model_folder_name)
    else:
        if args.model_path is None:
            raise ValueError("Vous devez fournir --model_path si --skip_training est activé")
        models_dir = os.path.dirname(args.model_path)
        model_dir = os.path.dirname(models_dir)
        logs_dir = os.path.join(model_dir, "logs")
        videos_dir = os.path.join(model_dir, "videos")
        model_name = "pose_model"
        model_path = args.model_path

    tflite_path = None  # Initialiser
    if not args.skip_data_prep:
        train_ds, val_ds = prepare_data()
        print("Data set created")
            
    
    if not args.skip_training:
        model = create_model()
        print("Model created")

        model_name = "pose_model"  # Nom simplifié car le dossier contient déjà la date/backbone
        history_back, history_head = train_model(model=model, tf_data_set=(train_ds, val_ds), model_name=model_name, model_dir=model_dir)
        print("Model trained")

        final_model_path, saved_model_dir = save_final_model(model, model_name, model_dir)
        print("Model_saved")
    
    if args.plot_history:
        name = ["backbone", "head"]
        for h, hist in enumerate([ history_back, history_head]):
            plot_path = os.path.join(logs_dir, f"{model_name}_{name[h]}_history.png")
            plot_training_history(hist, save_path=plot_path)
    
    # import tensorflow as tf
    # # model_path = r"C:\Users\neuromolity-lab\Documents\amedeo\pose-estimation-finetune\output\MNv3S_20251231_123442\models\pose_model_final.h5"
    # # model = tf.keras.models.load_model(model_path)
    # # model = tf.saved_model.load(saved_model_dir)
    # import tensorflow as tf
    # img, gt_heatmaps = next(iter(X_val.take(1)))
    # # img = img[8]
    # y_pred = model.predict(img)
    # print("logits:", tf.reduce_min(y_pred), tf.reduce_max(y_pred))
    # print("sigmoid max:", tf.reduce_max(tf.sigmoid(y_pred)))
    # sig = tf.sigmoid(y_pred).numpy()[0]
    # scale = config.INPUT_SHAPE[0] // config.HEATMAP_SIZE[0]
    # coords = [np.array(np.where(sig[:, :, i] == np.max(sig[:, :, i]))).flatten() * scale for i in range(sig.shape[-1])]
    # gt_coords = from_heatmaps_to_coords(gt_heatmaps, from_prediction=False)
    # pr_coords = from_heatmaps_to_coords(y_pred, from_prediction=True)
    # import matplotlib.pyplot as plt
    # import cv2
    # gt_coords = gt_coords[0]
    # pr_coords = pr_coords[0]
    # img = img[0]
    # img = img.numpy().astype(np.uint8)
    
    # heatmap = np.clip(y_pred[0], 0, 1)
    # heatmap = cv2.applyColorMap(
    #         (heatmap * 255).astype(np.uint8),
    #         cv2.COLORMAP_JET
    #     )
    # heatmaps = cv2.resize(heatmap, (int(img.shape[0]), int(img.shape[1])))
    # img = cv2.addWeighted(img,  0.5, heatmaps, 0.5, 0)
    # [plt.scatter(gt_coords[1, i], gt_coords[0, i], color="r") for i in range(gt_coords.shape[-1])]
    # [plt.scatter(pr_coords[1, i], pr_coords[0, i], color="b", s=10) for i in range(pr_coords.shape[-1])]
    # plt.imshow(img)

    if not args.skip_export:
        tflite_paths = export_model(model_path=saved_model_dir, model_name=model_name, model_dir=model_dir, representative_ds=val_ds)

    if args.test_tflite:
        test_tflite_model(model_path.replace("final.keras", "dynamic.tflite"), val_ds=val_ds, num_samples=10)

def parse_arguments():
    """
    Parse les arguments de la ligne de commande
    """
    parser = argparse.ArgumentParser(
        description="Pipeline de fine-tuning pour la pose estimation"
    )
    
    # Options de workflow
    parser.add_argument(
        '--skip-data-prep',
        action='store_true',
        help="Sauter la préparation des données (charge depuis le cache)"
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help="Sauter l'entraînement (utilise un modèle existant)"
    )
    parser.add_argument(
        '--skip-export',
        action='store_true',
        help="Sauter l'export TFLite"
    )
    
    # Configuration du modèle
    parser.add_argument(
        '--backbone',
        type=str,
        default="MobileNetV3Small",
        choices=[
            'MobileNetV2', 'MobileNetV3Small', 'MobileNetV3Large',
            'EfficientNetLite0', 'EfficientNetLite1', 'EfficientNetLite2', 
            'EfficientNetLite3', 'EfficientNetLite4',
            'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
            'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3'
        ],
        help="Backbone à utiliser (défaut: MobileNetV3Small)"
    )
    
    # Options de sauvegarde
    parser.add_argument(
        '--save-data',
        action='store_true',
        help="Sauvegarder les données prétraitées"
    )
    parser.add_argument(
        '--plot-history',
        action='store_true',
        default=True,
        help="Tracer les courbes d'apprentissage"
    )
    parser.add_argument(
        '--test-tflite',
        action='store_true',
        default=True,
        help="Tester le modèle TFLite après conversion"
    )
    
    # Chemins
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help="Chemin vers un modèle existant (si --skip-training)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    # Parser les arguments
    args = parse_arguments()
    # args.skip_training = True
    # # args.skip_export = True
    # args.model_path = r"/mnt/c/Users/neuromolity-lab/Documents/amedeo/pose-estimation-finetune/output/MNv3S_20260102_103228/models/pose_model_final.keras"

    # Lancer le pipeline
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n\n❌ Erreur lors de l'exécution du pipeline:")
        print(f"   {type(e).__name__}: {e}")
        raise
