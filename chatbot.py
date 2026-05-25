import tensorflow as tf
import numpy as np
import json
import os


def load_kb(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def predict_disease(model, img_path, class_names):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)

    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array, verbose=0)

    score = predictions[0]
    class_idx = np.argmax(score)

    return class_names[class_idx], 100 * np.max(score)


def run_chatbot():
    print("Expert System Initializing...")
    model = tf.keras.models.load_model('plant_disease_model.keras')
    kb = load_kb('knowledge-base.json')

    class_names = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
        "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
        "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
        "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
        "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
        "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]

    print("System Ready! Type 'exit' to quit.")

    while True:
        user_input = input("\nEnter the file path of the leaf image: ")

        if user_input.lower() == 'exit':
            break

        if not os.path.exists(user_input):
            print("File not found. Please try again.")
            continue

        try:
            disease, confidence = predict_disease(model, user_input, class_names)
            print(f"\nDiagnosis: {disease} (Confidence: {confidence:.2f}%)")

            if disease in kb:
                info = kb[disease]
                print("\nKnowledge Base Match Found:")
                print(f"Symptoms: {', '.join(info.get('symptoms', []))}")
                print(f"Recommended Treatment: {', '.join(info.get('treatment', []))}")
            else:
                print("\nNote: Disease recognized by vision model, but no details found in the Knowledge Base yet.")

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    run_chatbot()