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
from train import train_model, plot_training_history
from export_tflite import export_model, test_tflite_model
from validation_utils import  evaluate_model
import tensorflow as tf
from train_utils import LandmarkHuberLoss



def main(args):

    # Configurer le backbone si spécifié en argument
    if args.backbone:
        config.BACKBONE = args.backbone
    print(f"\nConfiguration:")
    print(f"   - Backbone: {config.BACKBONE}")
    print(f"   - Input size: {config.INPUT_SHAPE[0]}x{config.INPUT_SHAPE[1]}")

    overfit = args.overfit_one_frame
    if not args.skip_training:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_folder_name = config.get_model_folder_name(config.BACKBONE, timestamp)
        model_dir, models_dir, logs_dir, videos_dir = config.setup_model_directories(model_folder_name)
    else:
        if args.model_path is None:
            raise ValueError("Vous devez fournir --model_path si --skip_training est activé")
        models_dir = os.path.dirname(args.model_path)
        model_dir = os.path.dirname(models_dir)
        model_name = "pose_model"
        model_path = args.model_path
        logs_dir = os.path.join(model_dir, "logs")
        if not os.path.exists(logs_dir):
            logs_dir = os.path.join(os.path.dirname(model_dir), "logs")
    if args.checkpoints_dir:
        models_dir = os.path.dirname(args.checkpoints_dir)
        model_dir = os.path.dirname(models_dir)

    tflite_path = None  # Initialiser
    if not args.skip_data_prep:
        train_ds, val_ds = prepare_data(overfit)
        print("Data set created")
            
    if not args.skip_training:
        model = create_model(overfit)
        print("Model created")

        model_name = "pose_model"
        model = train_model(model=model, tf_data_set=(train_ds, val_ds), model_name=model_name, model_dir=model_dir, overfit=overfit)
        ds = val_ds if not overfit else train_ds
        evaluate_model(model, ds, model_dir)
    
    if args.skip_training:
        model = tf.keras.models.load_model(model_path) #, custom_objects={"LandmarkHuberLoss", LandmarkHuberLoss(56, 126, 1)},)
        plot_path = os.path.join(logs_dir, f"pose_model_loss.png")
        plot_training_history(save_path=None, csv_path=logs_dir)
        evaluate_model(model, val_ds, model_dir)

    if not args.skip_export:
        if args.skip_data_prep:
            train_ds, val_ds = prepare_data()
        tflite_paths = export_model(model = model, model_name=model_name, model_dir=model_dir, representative_ds=val_ds)

    if args.test_tflite:
        if args.skip_data_prep:
            train_ds, val_ds = prepare_data()
        test_tflite_model(models_dir, val_ds=val_ds, num_samples=20)

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
    parser.add_argument(
        '--plot-validation',
        action='store_true',
        help="plot image after prediction on the validation dataset"
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

    parser.add_argument(
        '--overfit-one-frame',
        default=False,
        help="Overfit the model on one image to try the training configuration."
    )

    # Chemins
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help="Chemin vers un modèle existant (si --skip-training)"
    )

    parser.add_argument(
        '--checkpoints-dir',
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
    args.backbone = "MobileNetV3Large" #"MobileNetV3Large"
    args.overfit_one_frame = False
    # args.skip_training = True
    # args.skip_export = True
    args.test_tflite = False
    # args.model_path = r"/mnt/c/Users/Usager/Documents/Amedeo/pose-estimation-finetune/output/ENL4_20260115_171548/models/checkpoints/model_backbone_checkpoint_3300.keras"
    # args.model_path = r"/mnt/c/Users/Usager/Documents/Amedeo/pose-estimation-finetune/output/MNv3L_20260120_134951/models"
    # args.checkpoints_dir = r"/mnt/c/Users/Usager/Documents/Amedeo/pose-estimation-finetune/output/MNv3L_20260120_134951/models/checkpoints"

    # Lancer le pipeline
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\nPipeline interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n\nErreur lors de l'exécution du pipeline:")
        print(f"   {type(e).__name__}: {e}")
        raise
