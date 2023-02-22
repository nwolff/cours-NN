import streamlit as st
import json
import requests
import matplotlib.pyplot as plt
import numpy as np

URI = "http://localhost:8888"

st.title("Visualiseur de Réseau de Neurones")
st.sidebar.markdown("Image d'entrée")

if st.button("Recevoir une prédiction aléatoire"):
    response = requests.post(URI, data={})
    response = json.loads(response.text)
    preds = response.get("prediction")
    image = response.get("image")
    image = np.array(image)

    st.sidebar.image(image, width=150)

    for layer, p in enumerate(preds):
        numbers = np.squeeze(np.array(p))
        fig = plt.figure(figsize=(32, 4))
        if layer == 2:
            row, col = 1, 10
        else:
            row, col = 2, 16
        for i, number in enumerate(numbers):
            plt.subplot(row, col, i + 1)
            plt.imshow(number * np.ones((8, 8, 3)).astype("float32"))
            plt.xticks([])
            plt.yticks([])

            if layer == 2:
                plt.xlabel(str(i), fontsize=40)

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()
        st.text("Couche {}".format(layer + 1))
        st.pyplot(fig)