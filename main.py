"""
Script principal pour le pipeline de fine-tuning
"""
import os
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

    # ÉTAPE 0: Configuration des dossiers
    print("\n📁 CONFIGURATION DES DOSSIERS")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_folder_name = config.get_model_folder_name(config.BACKBONE, timestamp)
    model_dir, models_dir, logs_dir, videos_dir = config.setup_model_directories(model_folder_name)

    print(f"📂 Dossier modèle: {model_folder_name}")
    print(f"   - Modèles: {models_dir}")
    print(f"   - Logs: {logs_dir}")
    print(f"   - Vidéos: {videos_dir}")

    tflite_path = None  # Initialiser

    # ÉTAPE 1: Préparation des données
    if not args.skip_data_prep:
        print("\nÉTAPE 1/4 - PRÉPARATION DES DONNÉES")
        X_train, X_val, y_train, y_val = prepare_data()

        if args.save_data:
            data_path = os.path.join(model_dir, "preprocessed_data.npz")
            np.savez_compressed(data_path, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
            print(f"💾 Données sauvegardées: {data_path}")
    else:
        print("\n⏩ Chargement des données prétraitées...")
        data_path = os.path.join(model_dir, "preprocessed_data.npz")
        data = np.load(data_path)
        X_train = data['X_train']
        X_val = data['X_val']
        y_train = data['y_train']
        y_val = data['y_val']
        print(f"✅ Données chargées depuis: {data_path}")
    
    # ÉTAPE 2: Construction du modèle
    if not args.skip_training:
        print("\nÉTAPE 2/4 - CONSTRUCTION DU MODÈLE")
        model = create_model()

        # ÉTAPE 3: Entraînement
        print("\nÉTAPE 3/4 - ENTRAÎNEMENT")
        model_name = "pose_model"  # Nom simplifié car le dossier contient déjà la date/backbone

        history = train_model(model=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, model_name=model_name, model_dir=model_dir)
        final_model_path, saved_model_dir = save_final_model(model, model_name, model_dir)
        metrics = evaluate_model(model, X_val, y_val)

        if args.plot_history:
            plot_path = os.path.join(logs_dir, f"{model_name}_history.png")
            plot_training_history(history, save_path=plot_path)
    else:
        print("\n⏩ Chargement du modèle entraîné...")
        model_path = args.model_path
        if not model_path:
            raise ValueError("Vous devez fournir --model_path si --skip_training est activé")
        saved_model_dir = model_path
        model_name = "pose_model"
        print(f"✅ Modèle chargé depuis: {saved_model_dir}")

    # ÉTAPE 4: Export TFLite
    tflite_paths = None
    if not args.skip_export:
        print("\nÉTAPE 4/4 - EXPORT TENSORFLOW LITE")
        tflite_paths = export_model(model_path=saved_model_dir, X_val=X_val, model_name=model_name, model_dir=model_dir)

        if args.test_tflite:
            # Tester le modèle recommandé (dynamic)
            test_tflite_model(tflite_paths['dynamic'], X_val, y_val, num_samples=10)
    
    # Résumé final
    print("\n" + "=" * 60)
    print("🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")
    print("=" * 60)
    print(f"\n📂 Résultats sauvegardés dans: {model_dir}")
    print(f"   - Modèles: {models_dir}")
    print(f"   - Logs: {logs_dir}")
    print(f"   - Vidéos: {videos_dir}")

    if tflite_paths:
        print(f"\n📱 Modèles TFLite prêts pour le déploiement:")
        print(f"   ⭐ PRODUCTION: {os.path.basename(tflite_paths['dynamic'])}")
        print(f"   🔬 TESTS: {os.path.basename(tflite_paths['float32'])}")
        print(f"\n💡 Prochaines étapes:")
        print(f"   1. Testez le modèle dynamic sur de nouvelles images")
        print(f"   2. Intégrez-le dans votre application mobile")
        print(f"   3. Utilisez GPU Delegate (Android) ou Metal Delegate (iOS) pour accélérer")

    print("\n" + "=" * 60)


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
    # Parser les arguments
    args = parse_arguments()
    
    # Lancer le pipeline
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n\n❌ Erreur lors de l'exécution du pipeline:")
        print(f"   {type(e).__name__}: {e}")
        raise
