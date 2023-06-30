import streamlit as st
import numpy as np
from PIL import Image
import pickle
import requests
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def main():
    # Categorical inputs
    
    
    image = Image.open('image.png')
    st.title("Car Price Predictor")

    st.image(image, caption=f"Deals On Wheels Pvt. Ltd.", use_column_width=True)
    st.subheader("Welcome to The Deals On Wheels Pvt. Ltd !")

    st.sidebar.subheader(
        "Own your dream car. This company is based on second hand or brand new cars.")
    st.sidebar.write("Prabhat Chaudhary ", "\n", "CEO")
    st.sidebar.subheader('Contact Us.  \n'
                         'Email:-  prabhatchaudhary8496@gmail.com')

    st.sidebar.subheader("+91-7398218897")

    Owner_type = {"First Owner": 1, "Second Owner": 2, "Third Owner": 3, "Fourth Owner and Above Owner": 4,
              "Test Drive Car": 5}
    Seller_type = {"Individual": 1, "Dealer": 2, "Trust mark Dealer": 3}
    Transmission1 = {"Manual": 1, "Automatic": 2}
    Brands = {"Maruti": 1, "Hyundai": 2, "Mahindra": 3, "Tata": 4, "Honda": 5, "Ford": 6, "Toyota": 7, "Chevrolet": 8,
              "Renault": 9, "Volkswagen": 10,
              "Skoda": 11, "Nissan": 12, "Audi": 13, "BMW": 14, "Fiat": 15, "Datsun": 16, "Mercedes-Benz": 17,
              "Jaguar": 18, "Mitsubishi": 19, "Land": 20,
              "Volvo": 21, "Ambassador": 22, "Jeep": 23, "MG": 24, "OpelCorsa": 25, "Daewoo": 26, "Force": 27,
              "Isuzu": 28, "Kia ": 29}
    Fuel_type = {"Diesel": 1, "Patrol": 2, "CNG": 3, "LPG": 4,"Electric": 5}
    #Car_model = st.selectbox('Car Origin', LabelEncoder.inverse_transform(df['encoded_origin'].unique()))


    #name = st.selectbox("Car model",Car_model)
    brand = st.selectbox("Brand", tuple(Brands.keys()))
    year = st.number_input("Year of purchase", 1900, 2023)
    driver = st.number_input("Driver(KM)")
    owner_type = st.selectbox("Owner Type", tuple(Owner_type.keys()))
    engine_type = st.selectbox("Fuel type", tuple(Fuel_type.keys()))
    transmission_type = st.selectbox("Transmission", tuple(Transmission1.keys()))
    seller_type = st.selectbox("Seller type", tuple(Seller_type.keys()))

    def get_value(val, my_dict):
        for key, value in my_dict.items():
            if val == key:
                return value

    def load_model(model_file):
        model = pickle.load(open(model_file, "rb"))
        return model

    if st.button("Predict"):
        feature_list = [get_value(brand, Brands), int(year), int(driver), get_value(owner_type, Owner_type),
                        get_value(engine_type, Fuel_type), get_value(transmission_type, Transmission1),
                        get_value(seller_type, Seller_type)]
        # st.write(feature_list)
        st.subheader("Your Input")
        user_input_data = {"Name": Car_model, "Brand": brand, "Year of purchase": year, "Drive(KM)": driver,
                           "Owner_Type": owner_type, "Engine Type": engine_type, "Transmission Type": transmission_type,
                           "Seller Type": seller_type}
        st.write(user_input_data)
        st.subheader("Predicted Selling Price")
        input_data = np.array(feature_list).reshape(1, -1)
        model =load_model("final_model.pkl")
        prediction = model.predict(input_data)
        st.write("Predicted Selling Price :" + " " + "₹" +" " + str(np.round(prediction[0], 2)))

        st.subheader(''' Thank you for your visit !''')


if __name__ == "__main__":
    main()