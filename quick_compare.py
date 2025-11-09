"""
Script de comparaison complète : Keras vs Dynamic vs Float32
"""
import numpy as np
from compare_models import compare_models
import os
from pathlib import Path

def find_latest_model(pattern):
    """Trouve le modèle le plus récent correspondant au pattern"""
    from config import OUTPUT_DIR
    output_dir = Path(OUTPUT_DIR)
    
    # Chercher dans tous les dossiers de modèles
    matching_models = []
    for model_dir in output_dir.iterdir():
        if model_dir.is_dir() and not model_dir.name.startswith('.'):
            models_subdir = model_dir / "models"
            if models_subdir.exists():
                matching_models.extend(list(models_subdir.glob(pattern)))
    
    if not matching_models:
        return None
    # Trier par date de modification (le plus récent en premier)
    return max(matching_models, key=lambda p: p.stat().st_mtime)

# Charger les données depuis le modèle le plus récent
print("📂 Chargement des données...")
if keras_model:
    # Utiliser les données du dossier du modèle Keras le plus récent
    model_dir = keras_model.parent.parent
    data_path = model_dir / "preprocessed_data.npz"
    if data_path.exists():
        data = np.load(str(data_path))
        X_val = data['X_val']
        print(f"✅ Données chargées depuis: {data_path}")
    else:
        print(f"❌ Données non trouvées dans: {data_path}")
        print("💡 Les données doivent être dans le dossier du modèle")
        exit(1)
else:
    print("❌ Aucun modèle Keras trouvé pour charger les données")
    exit(1)

print("\n🔍 Recherche des modèles disponibles...")
print("=" * 60)

# Trouver les modèles
keras_model = find_latest_model('pose_model_*_best.h5')
dynamic_model = find_latest_model('pose_model_dynamic.tflite')
float32_model = find_latest_model('pose_model_float32.tflite')

models_found = []
if keras_model and keras_model.exists():
    models_found.append(('Keras', str(keras_model)))
    print(f"✅ Keras: {keras_model.name}")
else:
    print("❌ Aucun modèle Keras trouvé")

if dynamic_model and dynamic_model.exists():
    models_found.append(('Dynamic', str(dynamic_model)))
    print(f"✅ Dynamic: {dynamic_model.name}")
else:
    print("❌ Modèle Dynamic non trouvé")

if float32_model and float32_model.exists():
    models_found.append(('Float32', str(float32_model)))
    print(f"✅ Float32: {float32_model.name}")
else:
    print("❌ Modèle Float32 non trouvé")

if len(models_found) < 2:
    print("\n❌ Pas assez de modèles pour faire des comparaisons")
    exit(1)

# Charger les données depuis le modèle Keras le plus récent
print("\n📂 Chargement des données...")
if keras_model:
    # Utiliser les données du dossier du modèle Keras le plus récent
    model_dir = keras_model.parent.parent
    data_path = model_dir / "preprocessed_data.npz"
    if data_path.exists():
        data = np.load(str(data_path))
        X_val = data['X_val']
        print(f"✅ Données chargées depuis: {data_path}")
    else:
        print(f"❌ Données non trouvées dans: {data_path}")
        print("💡 Les données doivent être dans le dossier du modèle")
        exit(1)
else:
    print("❌ Aucun modèle Keras trouvé pour charger les données")
    exit(1)

print(f"\n🧪 Test sur {len(X_val)} échantillons de validation")
print("=" * 60)

# Effectuer toutes les comparaisons possibles
results = {}
comparisons = []

# Comparer chaque paire de modèles
for i, (name1, path1) in enumerate(models_found):
    for j, (name2, path2) in enumerate(models_found):
        if i < j:  # Éviter les doublons
            print(f"\n🔄 Comparaison: {name1} vs {name2}")
            try:
                # Différentes stratégies selon les types de modèles
                if name1 == 'Keras' and name2 in ['Dynamic', 'Float32']:
                    # Keras vs TFLite : utiliser compare_models existante
                    result = compare_models(
                        keras_path=path1,
                        tflite_path=path2,
                        X_test=X_val,
                        num_samples=50
                    )
                elif name1 in ['Dynamic', 'Float32'] and name2 in ['Dynamic', 'Float32']:
                    # TFLite vs TFLite : adapter la logique
                    print("   📊 Comparaison TFLite vs TFLite (même logique que Keras vs TFLite)")
                    # Pour simplifier, on compare avec le modèle Keras comme référence
                    keras_ref = [p for n, p in models_found if n == 'Keras'][0]
                    result1 = compare_models(
                        keras_path=keras_ref,
                        tflite_path=path1,
                        X_test=X_val,
                        num_samples=25
                    )
                    result2 = compare_models(
                        keras_path=keras_ref,
                        tflite_path=path2,
                        X_test=X_val,
                        num_samples=25
                    )
                    # Calculer la différence entre les deux TFLite
                    result = {
                        'avg_distance': abs(result1['avg_distance'] - result2['avg_distance']),
                        'max_distance': max(result1['max_distance'], result2['max_distance']),
                        'avg_conf_diff': abs(result1['avg_conf_diff'] - result2['avg_conf_diff'])
                    }
                    print(f"   📊 Différence {name1} vs {name2}:")
                    print(f"      Distance moyenne: {result['avg_distance']:.4f}")
                    print(f"      Distance max: {result['max_distance']:.4f}")
                    print(f"      Différence confiance moyenne: {result['avg_conf_diff']:.4f}")
                else:
                    print(f"   ⏭️  Comparaison {name1} vs {name2} ignorée (même type)")
                    continue
                
                key = f"{name1}_vs_{name2}"
                results[key] = result
                comparisons.append((name1, name2, result))
                
            except Exception as e:
                print(f"❌ Erreur lors de la comparaison {name1} vs {name2}: {e}")
                continue

# Afficher le résumé final
print("\n" + "=" * 80)
print("� RÉSUMÉ COMPARATIF FINAL")
print("=" * 80)
print("<10px = EXCELLENT  |  <20px = BON  |  >20px = À AMÉLIORER")
print("-" * 80)
print("Comparaison              | Erreur moy. | Erreur max | Statut")
print("-" * 80)

for name1, name2, result in comparisons:
    comp_name = f"{name1:>8} vs {name2:<8}"
    avg_px = result['avg_distance'] * 192
    max_px = result['max_distance'] * 192
    
    if avg_px < 5:
        status = "✅ EXCELLENT"
    elif avg_px < 15:
        status = "🟡 BON"
    else:
        status = "🔴 À AMÉLIORER"
    
    print("<25")

print("=" * 80)

# Recommandations
print("\n💡 RECOMMANDATIONS:")
if len([r for _, _, r in comparisons if r['avg_distance'] * 192 < 5]) > 0:
    print("   ✅ Excellente précision - Tous les modèles sont utilisables")
if len([r for _, _, r in comparisons if r['avg_distance'] * 192 > 15]) > 0:
    print("   ⚠️  Certains modèles ont une précision perfectible")
    print("   🔧 Considérer l'amélioration de la calibration ou le QAT")

print("\n🎯 MODÈLE RECOMMANDÉ POUR LA PRODUCTION:")
dynamic_results = [r for n1, n2, r in comparisons if 'Dynamic' in [n1, n2]]
if dynamic_results:
    avg_dynamic_error = np.mean([r['avg_distance'] * 192 for r in dynamic_results])
    if avg_dynamic_error < 10:
        print("   ⭐ TFLite Dynamic - Parfait compromis précision/taille ⚡")
    else:
        print("   🔬 TFLite Float32 - Meilleure précision mais plus volumineux")
else:
    print("   🔬 TFLite Float32 - Modèle de référence")

print("=" * 80)
