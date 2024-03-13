import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
import pickle
import os

model_Disease = tf.keras.models.load_model('plant_disease_model.h5')

class_labels = ['Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy', 'Blueberry_healthy',
                'Cherry_Powdery_mildew', 'Cherry_healthy', 'Corn(maize) Cercospora_leaf_spot Gray_leaf_spot',
                'Corn(maize) Common_rust', 'Corn_(maize) Northern_Leaf_Blight', 'Corn(maize) healthy',
                'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape_Leaf_blight (Isariopsis_Leaf_Spot)',
                'Grape_healthy', 'Orange_Haunglongbing (Citrus_greening)', 'Peach_Bacterial_spot', 'Peach_healthy',
                'Pepper,bell_Bacterial_spot', 'Pepper,bell_healthy', 'Potato_Early_blight', 'Potato_Late_blight',
                'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew',
                'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 'Tomato__Early_blight',
                'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot',
                'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato_healthy']

class_labels_hindi = ["सेब की छाल", "सेब की काली बीमारी", "सेब की छाल", 'स्वस्थ सेब', 'स्वस्थ नीलबदरी',
                     "चेरी चूर्णिल ओसिता", 'स्वस्थ चेरी', "मक्का की धूसर पत्ती बीमारी", "मक्का की आम धूसर बीमारी",
                     "मक्का की उत्तरी पत्ती बीमारी", "स्वस्थ मक्का", "अंगूर की काली बीमारी", "अंगूर की एस्का (काली खसरा)",
                     "अंगूर की पत्ती बीमारी", "स्वस्थ अंगूर", "संतरे की हौंगलॉन्गबिंग (सिट्रस ग्रीनिंग)",
                     "आड़ू की बैक्टीरियल स्पॉट", "स्वस्थ आड़ू", "शिमला मिर्च की बैक्टीरियल स्पॉट",
                     "स्वस्थ शिमला मिर्च", "आलू की पहली धूसर बीमारी", "आलू की दूसरी धूसर बीमारी", "स्वस्थ आलू",
                     "स्वस्थ रसभरी", "स्वस्थ सोयाबीन", "कद्दू की धूली बीमारी", "स्ट्रॉबेरी की पत्ती जलन",
                     "स्वस्थ स्ट्रॉबेरी", "टमाटर की बैक्टीरियल स्पॉट", "टमाटर की पहली धूसर बीमारी",
                     "टमाटर की दूसरी धूसर बीमारी", "टमाटर की पत्ती की फफूंदी", "टमाटर की सेप्टोरिया पत्ती बीमारी",
                     "टमाटर की मकड़ी माइट", "टमाटर की लक्ष्य स्थल", "टमाटर की पीली पत्ती मुड़ी वायरस",
                     "टमाटर की मोजेक वायरस", "स्वस्थ टमाटर"]

supplement_labels = ['Katyayani Prozol Propiconazole 25/% /EC Systematic Fungicide', 'Magic FungiX For Fungal disease',
                     'Katyayani All in 1 Organic Fungicide', 'Tapti Booster Organic Fertilizer', 'GreenStix Fertilizer',
                     'ROM Mildew Clean', 'Plantic Organic BloomDrop Liquid Plant Food ',
                     'ANTRACOL FUNGICIDE', '3 STAR M45 Mancozeb 75% WP Contact Fungicide',
                     'QUIT (Carbendazim 12% + Mancozeb 63% WP) Protective And Curative Fungicide',
                     'Biomass Lab Sampoorn Fasal Super', 'Captain 55% + Hexaconazole 5% WP Systematic Fungicide',
                     'Fonate 50% WP Foliar Spray Fungicide', 'TranceBud 720 (Propiconazole 25% EC)',
                     'Chequer Fungicide', 'HRD TRUSTPLUS BIO-FERTILIZER', 'Optimax Bio 50%, Bio Fungicide',
                     'Jeevamrut Organic Manure', 'Fertogro Seaweed Extract Powder', 'Fertogro Humic Acid 95% Powder',
                     'Fertogro Organic Potash Granules', 'Ghanshyam Organic Potash Powder',
                     'MUSCLE ORGANIC CALCIUM  LIQUID', 'SATTA (Nitrogen) - Organic Liquid Fertilizer',
                     'SAC-Bio Fungicide (Botanical extracts) ', 'Magic Soil Plant Growth Promoter',
                     'Water soluble fertilizer NPK 19:19:19', 'Organic Nutrient Enricher',
                     'Water Soluble Micronutrient Mixture ', 'Bio Organic Manure',
                     'BASMATI BIO FERTILIZER & PESTICIDES', 'Chemical fertilizer',
                     'Ammonium Nitrate(34% NH3)', 'Urea(46%N)', 'Mono Ammonium Phosphate (12% N, 61% P2O5)',
                     'Single Super Phosphate (16% P2O5)', 'Muriate of Potash (60% K2O)', 'Sulphate of Potash (50% K2O)',
                     'Diammonium Phosphate (18-46-0)', 'Potassium Chloride (60% K2O)',
                     'Urea Ammonium Nitrate (28% N, 35% K2O)', '20-20-0', 'NPK (10-26-26)', 'NPK (20-20-0)',
                     'NPK (13-40-13)', 'NPK (28-28-0)', 'NPK (15-15-15)', 'NPK (0-52-34)', 'NPK (10-26-26)',
                     'NPK (13-00-45)', 'NPK (13-00-45)', 'NPK (00-00-50)', 'NPK (00-00-50)']

def preprocess_image(image_data):
    img = image.load_img(image_data, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale by 1/255
    return img_array

def predict_disease(img_path):
    img_array = preprocess_image(img_path)
    prediction = model_Disease.predict(img_array)
    predicted_class_indices = np.argmax(prediction, axis=1)
    return predicted_class_indices[0]

def main():
    st.title("Plant Disease Detection")
    st.write("This app detects the disease present in a plant image.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file,width=150, caption='Uploaded Image.')
        st.write("")
        st.write("Result....")
        label = predict_disease(uploaded_file)
        st.write(f"The plant is suffering from: {class_labels[label]}")
        st.write(f"The plant is suffering from: {class_labels_hindi[label]}")
        st.write(f"Supplements suggested: {supplement_labels[label]}")
if __name__ == '__main__':
    main()
