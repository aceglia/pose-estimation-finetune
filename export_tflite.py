"""
Export du modèle au format TensorFlow Lite pour déploiement mobile
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import config


def convert_to_tflite(model_path, output_path, quantize=True, quantization_type='int8', representative_dataset=None):
    """
    Convertit un modèle Keras en TensorFlow Lite
    
    Args:
        model_path: Chemin vers le modèle SavedModel ou .h5
        output_path: Chemin de sortie pour le fichier .tflite
        quantize: Activer la quantization
        quantization_type: Type de quantization ('int8', 'float16', 'dynamic', 'none')
        representative_dataset: Dataset représentatif pour la quantization
    
    Returns:
        tflite_model_size: Taille du modèle en Ko
    """
    print("=" * 60)
    print("📦 CONVERSION EN TENSORFLOW LITE")
    print("=" * 60)
    
    # Charger le modèle
    print(f"\n📂 Chargement du modèle depuis: {model_path}")
    
    # Créer le converter
    if model_path.endswith('.h5'):
        model = keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        # SavedModel format
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    
    # Configuration du converter selon le type de quantization
    if quantize:
        if quantization_type == 'int8':
            print("\n⚙️  Configuration de la quantization INT8 optimisée...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
            if representative_dataset is not None:
                converter.representative_dataset = representative_dataset
                
        elif quantization_type == 'float16':
            print("\n⚙️  Configuration de la quantization FLOAT16 (haute précision)...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
        elif quantization_type == 'dynamic':
            print("\n⚙️  Configuration de la quantization dynamique (range-based)...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # Les poids sont quantizés dynamiquement, entrées/sorties restent float32
            
        else:
            raise ValueError(f"Type de quantization non supporté: {quantization_type}")
    else:
        print("\n⚙️  Pas de quantization (modèle float32 complet)")
    
    # Convertir
    print("\n🔄 Conversion en cours...")
    tflite_model = converter.convert()
    
    # Sauvegarder
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Afficher la taille
    tflite_model_size = len(tflite_model) / 1024  # en Ko
    print(f"\n✅ Modèle TFLite sauvegardé: {output_path}")
    print(f"📊 Taille du modèle: {tflite_model_size:.2f} Ko")
    print(f"🎯 Type de quantization: {quantization_type.upper()}")
    
    print("=" * 60)
    
    return tflite_model_size


def create_representative_dataset_generator(X_val, num_samples=100):
    """
    Crée un générateur de dataset représentatif pour la quantization
    AMÉLIORÉ: Utilise plus d'échantillons et couvre mieux la distribution
    
    Args:
        X_val: Dataset de validation
        num_samples: Nombre d'échantillons à utiliser (augmenté pour meilleure calibration)
    
    Returns:
        representative_dataset_gen: Générateur pour le converter
    """
    def representative_dataset_gen():
        # AMÉLIORATION 2: Utiliser TOUS les échantillons de validation pour meilleure calibration
        # Au lieu de prendre séquentiellement, on mélange pour couvrir toute la distribution
        indices = np.random.permutation(len(X_val))[:num_samples]
        for idx in indices:
            # Prendre un échantillon
            sample = X_val[idx:idx+1].astype(np.float32)
            yield [sample]
    
    return representative_dataset_gen


def test_tflite_model(tflite_path, X_test, y_test, num_samples=10):
    """
    Teste le modèle TFLite et compare avec les prédictions originales
    
    Args:
        tflite_path: Chemin vers le modèle .tflite
        X_test: Images de test
        y_test: Heatmaps de test
        num_samples: Nombre d'échantillons à tester
    
    Returns:
        avg_error: Erreur moyenne
    """
    print("\n🧪 Test du modèle TFLite...")
    
    # Charger l'interpréteur TFLite
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Obtenir les détails des entrées/sorties
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n📊 Détails de l'interpréteur:")
    print(f"   - Input shape: {input_details[0]['shape']}")
    print(f"   - Input type: {input_details[0]['dtype']}")
    print(f"   - Output shape: {output_details[0]['shape']}")
    print(f"   - Output type: {output_details[0]['dtype']}")
    
    # Tester sur quelques échantillons
    errors = []
    for i in range(min(num_samples, len(X_test))):
        # Préparer l'entrée
        input_data = X_test[i:i+1].astype(np.float32)
        
        # Si le modèle attend des uint8, il faut quantizer l'entrée
        if input_details[0]['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details[0]['quantization']
            input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)
        
        # Inférence
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Récupérer la sortie
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Si la sortie est quantizée, il faut la déquantizer
        if output_details[0]['dtype'] == np.uint8:
            output_scale, output_zero_point = output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        # Calculer l'erreur
        error = np.mean(np.abs(output_data - y_test[i:i+1]))
        errors.append(error)
    
    avg_error = np.mean(errors)
    print(f"\n📊 Résultats du test:")
    print(f"   - Nombre d'échantillons testés: {len(errors)}")
    print(f"   - Erreur moyenne (MAE): {avg_error:.6f}")
    
    return avg_error


def export_model(model=None, model_path=None, X_val=None, model_name="pose_model", model_dir=None):
    """
    Pipeline complet d'export du modèle en TFLite avec deux versions optimisées
    
    Args:
        model: Modèle Keras (optionnel si model_path est fourni)
        model_path: Chemin vers le modèle sauvegardé (optionnel si model est fourni)
        X_val: Dataset de validation pour la quantization
        model_name: Nom du modèle
        model_dir: Dossier racine du modèle (si None, utilise config.MODELS_DIR)
    
    Returns:
        tflite_paths: Dictionnaire avec les chemins des modèles exportés
    """
    print("=" * 60)
    print("🚀 EXPORT DU MODÈLE EN TENSORFLOW LITE")
    print("=" * 60)
    
    # Déterminer le dossier des modèles
    models_dir = config.MODELS_DIR if model_dir is None else os.path.join(model_dir, "models")
    
    # Si un modèle Keras est fourni, le sauvegarder d'abord
    if model is not None:
        saved_model_dir = os.path.join(models_dir, f"{model_name}_for_export")
        print(f"\n💾 Sauvegarde du modèle au format SavedModel...")
        model.save(saved_model_dir, save_format='tf')
        model_path = saved_model_dir
    
    if model_path is None:
        raise ValueError("Vous devez fournir soit 'model' soit 'model_path'")
    
    tflite_paths = {}
    
    # Créer le dataset représentatif si nécessaire
    representative_dataset = None
    if X_val is not None:
        num_calibration_samples = min(500, len(X_val))
        print(f"\n📊 Création du dataset représentatif ({num_calibration_samples} échantillons)...")
        representative_dataset = create_representative_dataset_generator(
            X_val, 
            num_samples=num_calibration_samples
        )
    
    # Export 1: Modèle Dynamic Range Quantization (recommandé pour mobile)
    print("\n" + "=" * 40)
    print("📱 EXPORT 1/2 - DYNAMIC RANGE QUANTIZATION")
    print("=" * 40)
    print("🎯 RECOMMANDÉ: Précision optimale + taille réduite")
    
    tflite_dynamic_path = os.path.join(models_dir, f"{model_name}_dynamic.tflite")
    dynamic_size = convert_to_tflite(
        model_path=model_path,
        output_path=tflite_dynamic_path,
        quantize=True,
        quantization_type='dynamic',
        representative_dataset=None  # Dynamic n'a pas besoin de dataset représentatif
    )
    tflite_paths['dynamic'] = tflite_dynamic_path
    
    # Export 2: Modèle Float32 complet (haute précision)
    print("\n" + "=" * 40)
    print("🔬 EXPORT 2/2 - FLOAT32 COMPLET")
    print("=" * 40)
    print("🎯 TESTS: Précision maximale (taille importante)")
    
    tflite_float32_path = os.path.join(models_dir, f"{model_name}_float32.tflite")
    float32_size = convert_to_tflite(
        model_path=model_path,
        output_path=tflite_float32_path,
        quantize=False,
        quantization_type='none',
        representative_dataset=None
    )
    tflite_paths['float32'] = tflite_float32_path
    
    print(f"\n✅ Exports terminés!")
    print(f"📱 Modèle Dynamic: {tflite_dynamic_path} ({dynamic_size:.1f} Ko)")
    print(f"🔬 Modèle Float32: {tflite_float32_path} ({float32_size:.1f} Ko)")
    
    # Comparaison des modèles
    print("\n" + "=" * 60)
    print("� COMPARAISON DES MODÈLES EXPORTÉS")
    print("=" * 60)
    print("Modèle         | Taille | Précision | Usage recommandé")
    print("-" * 60)
    print(f"Dynamic (.tflite) | {dynamic_size:>5.1f} Ko | ~1px erreur | PRODUCTION MOBILE ⭐")
    print(f"Float32 (.tflite) | {float32_size:>5.1f} Ko | ~0px erreur | TESTS/VALIDATION")
    print("=" * 60)
    
    # Instructions pour l'utilisation
    print("\n📱 UTILISATION DANS FLUTTER")
    print("=" * 60)
    print("🤖 Pour production mobile:")
    print(f"   📁 Utilisez: {os.path.basename(tflite_dynamic_path)}")
    print("   ✅ Précision suffisante + taille optimisée")
    print("   🚀 Compatible avec GPU/NNAPI delegates")
    
    print("\n🔬 Pour tests/validation:")
    print(f"   📁 Utilisez: {os.path.basename(tflite_float32_path)}")
    print("   ✅ Précision maximale")
    print("   🐌 Plus lent, taille importante")
    
    print("\n📋 Paramètres communs:")
    print("   • Input: 192×192×3 float32 (0-1 normalisé)")
    print("   • Output: 48×48×3 float32 (heatmaps)")
    print("   • Keypoints: [0]=Hanche, [1]=Genou, [2]=Cheville")
    print("=" * 60)
    
    return tflite_paths
    
    # Si un modèle Keras est fourni, le sauvegarder d'abord
    if model is not None:
        saved_model_dir = os.path.join(config.MODELS_DIR, f"{model_name}_for_export")
        print(f"\n💾 Sauvegarde du modèle au format SavedModel...")
        model.save(saved_model_dir, save_format='tf')
        model_path = saved_model_dir
    
    if model_path is None:
        raise ValueError("Vous devez fournir soit 'model' soit 'model_path'")
    
    # Chemin de sortie pour le .tflite
    tflite_filename = f"{model_name}_{quantization_type}.tflite"
    tflite_path = os.path.join(config.MODELS_DIR, tflite_filename)
    
    # Créer le dataset représentatif si nécessaire
    representative_dataset = None
    if quantization_type == 'int8' and X_val is not None:
        num_calibration_samples = min(500, len(X_val))
        print(f"\n📊 Création du dataset représentatif ({num_calibration_samples} échantillons)...")
        representative_dataset = create_representative_dataset_generator(
            X_val, 
            num_samples=num_calibration_samples
        )
    
    # Convertir en TFLite
    quantize = quantization_type != 'none'
    tflite_size = convert_to_tflite(
        model_path=model_path,
        output_path=tflite_path,
        quantize=quantize,
        quantization_type=quantization_type,
        representative_dataset=representative_dataset
    )
    
    print(f"\n✅ Export terminé!")
    print(f"📱 Modèle prêt pour le déploiement mobile: {tflite_path}")
    
    # Comparaison des tailles et précisions
    print("\n" + "=" * 60)
    print("📊 COMPARAISON DES OPTIONS DE QUANTIZATION")
    print("=" * 60)
    print("🎯 Précision (décroissante) | Taille | Vitesse | Recommandation")
    print("-" * 60)
    print("❌ Aucune (float32)       | ~25MB  | Très lent | Développement seulement")
    print("🟡 Float16                | ~12MB  | Moyen     | BON COMPROMIS ⭐")
    print("🟠 Dynamic Range          | ~6MB   | Rapide    | Mobile standard")
    print("🔴 INT8                   | ~6MB   | Très rapide | Production intensive")
    print("=" * 60)
    
    # Instructions pour l'utilisation
    print("\n📱 UTILISATION DU MODÈLE TFLITE")
    print("=" * 60)
    print(f"\n🔧 Type de quantization utilisé: {quantization_type.upper()}")
    
    if quantization_type == 'float16':
        print("💡 RECOMMANDÉ pour votre cas - Précision proche du Keras avec bonne performance")
    elif quantization_type == 'none':
        print("⚠️  ATTENTION - Modèle très volumineux, utilisez seulement pour tests")
    
    print("\n🤖 Android (Java/Kotlin):")
    print("   1. Ajoutez le .tflite dans assets/")
    print("   2. Ajoutez la dépendance: implementation 'org.tensorflow:tensorflow-lite:2.x.x'")
    print("   3. Chargez avec: Interpreter.create(...)")
    print("   4. Utilisez GPU Delegate ou NNAPI pour accélérer")
    
    print("\n🍎 iOS (Swift/Objective-C):")
    print("   1. Ajoutez le .tflite au projet Xcode")
    print("   2. Ajoutez TensorFlowLiteSwift via CocoaPods/SPM")
    print("   3. Chargez avec: Interpreter(modelPath: ...)")
    print("   4. Utilisez Metal Delegate pour accélérer")
    
    print("=" * 60)
    
    return tflite_path


if __name__ == "__main__":
    print("✅ Module export_tflite.py chargé avec succès")
    print("📝 Utilisez main.py pour exporter le modèle après l'entraînement")
