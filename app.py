import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

# Configuration de la page
st.set_page_config(
    page_title="Détection encre ferrogallique",
    page_icon="🔬",
    layout="wide"
)

# Titre
st.title("🔬 Détection encre ferrogallique - Roboflow")
st.markdown("---")

# Récupérer la clé API depuis les secrets
try:
    api_key = st.secrets["ROBOFLOW_API_KEY"]
except:
    st.error("❌ Clé API manquante ! Configurez ROBOFLOW_API_KEY dans les secrets Streamlit")
    st.info("👉 Allez dans Settings → Secrets sur Streamlit Cloud")
    st.stop()

# Initialiser le client Roboflow
try:
    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key
    )
    st.sidebar.success("✅ Connecté à Roboflow")
except Exception as e:
    st.error(f"❌ Erreur de connexion : {e}")
    st.stop()

# Configuration dans la sidebar
st.sidebar.header("⚙️ Configuration")

# Sélection du modèle
model_id = st.sidebar.selectbox(
    "Choisir le modèle Roboflow",
    [
        "encre-ferrogallique-2-wy9md/5",
        "encre-ferrogallique-2-wy9md/3",
        "encre-ferrogallique-2-wy9md/2"
    ],
    index=0,
    help="Sélectionnez la version du modèle"
)

# Seuil de confiance
seuil_confiance = st.sidebar.slider(
    "Seuil de confiance",
    min_value=0.0,
    max_value=1.0,
    value=0.39,
    step=0.01,
    help="Détections en dessous de ce seuil seront ignorées"
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Modèle actif :**\n`{model_id}`\n\n**Seuil :** {seuil_confiance:.2f}")

# Section principale - Upload
st.header("📤 Télécharger une image")

uploaded_file = st.file_uploader(
    "Choisissez une image de manuscrit",
    type=["jpg", "jpeg", "png"],
    help="Formats acceptés : JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Créer deux colonnes
    col1, col2 = st.columns(2)
    
    # Afficher l'image originale
    with col1:
        st.subheader("📄 Image originale")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    # Sauvegarder temporairement
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        tmp_path = tmp_file.name
    
    # Lancer la détection
    try:
        with st.spinner("🔍 Analyse en cours..."):
            result = CLIENT.infer(tmp_path, model_id=model_id)
        
        # Charger l'image avec OpenCV
        img = cv2.imread(tmp_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Filtrer les prédictions
        predictions_filtered = [
            p for p in result["predictions"] 
            if p["confidence"] >= seuil_confiance
        ]
        
        # Dessiner les boîtes de détection
        for pred in predictions_filtered:
            x, y, width, height = pred["x"], pred["y"], pred["width"], pred["height"]
            x1, y1 = int(x - width/2), int(y - height/2)
            x2, y2 = int(x + width/2), int(y + height/2)
            
            label = f"{pred['class']} {pred['confidence']:.2f}"
            
            # Rectangle rouge épais
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
            # Fond pour le texte
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                img_rgb, 
                (x1, y1 - text_h - 10), 
                (x1 + text_w, y1), 
                (255, 0, 0), 
                -1
            )
            
            # Texte blanc
            cv2.putText(
                img_rgb, label, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        
        # Afficher l'image annotée
        with col2:
            st.subheader("🎯 Détections")
            st.image(img_rgb, use_container_width=True)
        
        # Résumé des détections
        st.markdown("---")
        st.subheader(f"📊 Résultats : {len(predictions_filtered)} détection(s)")
        
        if predictions_filtered:
            # Afficher chaque détection
            for i, pred in enumerate(predictions_filtered, 1):
                x, y, width, height = pred["x"], pred["y"], pred["width"], pred["height"]
                
                with st.expander(f"🔍 Détection #{i} - {pred['class']} ({pred['confidence']:.1%})"):
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Classe", pred['class'])
                    with col_b:
                        st.metric("Confiance", f"{pred['confidence']:.1%}")
                    with col_c:
                        st.metric("Dimensions", f"{int(width)}×{int(height)} px")
                    
                    st.caption(f"Position : ({int(x)}, {int(y)})")
        else:
            st.warning(f"⚠️ Aucune détection avec confiance ≥ {seuil_confiance:.2f}")
            st.info("💡 Essayez de réduire le seuil de confiance")
        
        # Données brutes (optionnel)
        with st.expander("🔧 Voir les données JSON brutes"):
            st.json(result)
        
    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse : {e}")
        st.info("Vérifiez que votre clé API et le modèle sont corrects")
    
    finally:
        # Nettoyer le fichier temporaire
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

else:
    # Instructions si aucune image
    st.info("👆 Téléchargez une image pour commencer l'analyse")
    
    with st.expander("ℹ️ Comment utiliser cette application"):
        st.markdown("""
        ### Mode d'emploi :
        
        1. **Configurez le modèle** dans la barre latérale (gauche)
        2. **Ajustez le seuil** de confiance si nécessaire
        3. **Téléchargez une image** de manuscrit ancien
        4. **Visualisez les résultats** avec les zones d'encre ferrogallique détectées
        
        ### À propos :
        
        Cette application utilise l'intelligence artificielle pour détecter 
        les zones d'encre ferrogallique dans les manuscrits anciens.
        
        L'encre ferrogallique était couramment utilisée du Moyen Âge jusqu'au 
        XXe siècle et peut causer des dégradations du papier avec le temps.
        """)
