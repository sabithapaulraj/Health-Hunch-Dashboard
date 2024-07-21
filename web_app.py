import os
import streamlit as st
import pandas as pd
import joblib
import plotly_express as px
from PIL import Image
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Health Hunch",
                   layout="wide",
                   page_icon="heart-pulse-fill")

# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

model = joblib.load("pickle files/model.pkl")
symptom_encoders = joblib.load("pickle files/symptom_encoder.pkl")
disease_encoder = joblib.load("pickle files/disease_encoder.pkl")
X_columns = joblib.load("pickle files/X_column.pkl")

image = Image.open("C:/Users/annbl/OneDrive/Documents/IBM_proj/logo.png")
col1, col2 = st.columns([0.4,0.5])
with col1:
    st.image(image,width=350)

html_title = """
    <style>
    .title-test {
    font-weight:bold;
    font-style:Cambria;
    padding:7px;
    border-radius:6px;
    }
    </style>
    <h1 class="title-test"> Stay ahead of the Curve</h1>
    <h4 class="title-test"> Predict, Prevent, Prosper ...</h4>"""
with col2:
    st.markdown(html_title, unsafe_allow_html=True)

#st.markdown("## Stay ahead of the Curve")
#st.markdown("##### Predict, Prevent, Prosper ...")

st.markdown("""---""")

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Health Hunch',
                           ['About',
                            'Understanding our Data',
                            'Self-Diagnosis'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'clipboard2-data-fill', 'graph-up-arrow'],
                           default_index=0)

def clean_data(df, df1):
    col = df.columns
    data = df[col].values.flatten()

    s = pd.Series(data)
    s = s.str.strip()
    s = s.values.reshape(df.shape)

    df = pd.DataFrame(s, columns=df.columns)
    df = df.fillna(0)
    
    col = df.columns
    vals = df.values
    symptoms = df1['Symptom'].unique()

    for i in range(len(symptoms)):
        vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]

    d = pd.DataFrame(vals, columns=col)
    d = d.replace('dischromic _patches', 0)
    d = d.replace('spotting_ urination', 0)
    df_cleaned = d.replace('foul_smell_of urine', 0)
    
    return df_cleaned

def visualize_data(df_cleaned):
    col = df_cleaned.columns
    # Plotting symptom counts
    symptoms = df_cleaned.drop(columns=['Disease'])
    symptom_counts = symptoms.apply(pd.value_counts).sum(axis=1)
    symptom_counts = symptom_counts.sort_values(ascending=False)

    fig1 = px.bar(symptom_counts, 
                  x=symptom_counts.values, 
                  y=symptom_counts.index, 
                  labels={'x': 'Count', 'y': 'Symptoms'}, 
                  title='Top Symptoms Counts', 
                  orientation='h')
    
    # Visualizing the distribution of symptoms across the dataset
    symptoms_stack = symptoms.stack().value_counts()
    fig2 = px.bar(symptoms_stack, 
                  x=symptoms_stack.index, 
                  y=symptoms_stack.values, 
                  labels={'x': 'Symptoms', 'y': 'Count'}, 
                  title='Distribution of Symptoms')
    
    disease_counts = df_cleaned['Disease'].value_counts()
    
    fig3 = px.pie(disease_counts, 
                  values=disease_counts.values, 
                  names=disease_counts.index, 
                  title='Distribution of Diseases')
    
    # Box plot of symptoms by disease
    df_melted = pd.melt(df_cleaned, id_vars=['Disease'], value_vars=df_cleaned.columns[1:], 
                        var_name='Symptom', value_name='Presence')
    
    fig4 = px.box(df_melted, 
                  x='Disease', 
                  y='Presence', 
                  color='Disease', 
                  points='all', 
                  title='Symptom Distribution by Disease')
    
    # Heatmap of symptom correlation
    symptoms_corr = df_cleaned.drop(columns=['Disease']).corr()

    fig5 = px.imshow(symptoms_corr,
                     labels=dict(x="Symptoms", y="Symptoms", color="Correlation"),
                     x=symptoms_corr.index,
                     y=symptoms_corr.columns,
                     title='Symptom Correlation Heatmap')
    
    return fig1, fig2, fig3, fig4, fig5

def predict_disease(symptoms):
    new_data = pd.DataFrame([symptoms], columns=X_columns)

    # Encode new instance using pre-fitted encoders
    for column in new_data.columns:
        if column in symptom_encoders:
            encoder = symptom_encoders[column]
            # Handle unknown labels with a default value
            new_data[column] = new_data[column].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

    predicted_label = model.predict(new_data)
    predicted_disease = disease_encoder.inverse_transform(predicted_label)[0]

    return predicted_disease


if selected == "About":
    #image = Image.open("Images/Landing Page Clip.gif")
    st.image("Images/Landing Page Clip.gif")
    
    st.markdown(""" """)
    st.markdown(""" """)
    st.markdown(""" """)
    st.title("About")
    col5, col6 = st.columns([1.5, 1.3])
    col3, col4 = st.columns([2, 3])
    col7, col8 = st.columns([0.9, 0.5])
    col9, col10 = st.columns([0.3, 0.9])
    col11, col12 = st.columns([0.9, 0.5])
    col13, col14 = st.columns([0.3, 0.9])
    
    with col5:
        st.markdown("##### Welcome to Health Hunch, a Symptom Analysis and Disease Prediction Dashboard, your comprehensive tool for understanding and managing health symptoms and conditions. This dashboard is designed to help individuals gain insights into various diseases and their symptoms, empowering you with the knowledge to make informed decisions about your health.")
    with col6:
        st.image("Images/about.gif", use_column_width=True)
        st.markdown(""" """)
        st.markdown(""" """)

    with col3:
        st.image("Images/Self Protection.gif", use_column_width=True)
        st.markdown(""" """)
    with col4:
        st.header("Key Features")
        st.markdown("""
                    - **Symptom Checker:** Easily search for symptoms by selecting the first letter of the symptom. Get detailed descriptions and information about each symptom to better understand potential health issues.
                    - **Disease Information:** Access an extensive database of diseases and conditions. Learn about the causes, symptoms, and treatment options for various health conditions.
                    - **User-Friendly Interface:** The dashboard is designed with a clean, intuitive interface, making it easy for users of all ages to navigate and find the information they need.
                    - **Visual Analytics:** Utilize visual analytics to see trends and patterns in symptoms and diseases. This can help in identifying common health issues and understanding the spread and prevalence of certain conditions.""")
        st.markdown(""" """)
                    
    st.markdown(""" """)
    st.markdown(""" """)
    
    with col7:
        st.header("How It Works")
        st.markdown("""**Search Symptoms:** Use the alphabetical buttons to search for symptoms by their first letter. Click on any letter to see a list of symptoms starting with that letter along with detailed descriptions.""")
        st.markdown("""
                    - Our Symptom Checker allows you to quickly and easily search for symptoms by their first letter. The intuitive alphabetical navigation helps you locate symptoms without needing to know their exact spelling.
                    - Once you click on a letter, you will be presented with a list of symptoms that start with that letter. Each symptom is accompanied by a detailed description, including common causes and associated conditions.
                    - This feature is particularly useful for those experiencing multiple or vague symptoms, as it helps narrow down potential health issues and provides a starting point for further research or consultation with healthcare professionals.""")
        st.markdown(""" """)
    with col8:
        st.image("Images/Search.gif", use_column_width=True)
    
    with col9:
        st.image("Images/Explore_Diseases.jpg", use_column_width=True)
    with col10:
        st.markdown("""**Explore Diseases:** Access detailed information on a wide range of diseases and conditions. Understand the symptoms, causes, and treatment options for each condition.""")
        st.markdown("""
                    - Our extensive disease database offers detailed information on a wide range of diseases and conditions. This resource is invaluable for anyone looking to understand the complexities of various health issues.
                    - For each disease, you will find comprehensive information on symptoms, causes, risk factors, and treatment options. Additionally, we provide insights into prognosis and preventive measures, helping you make informed decisions about your health. """)
        st.markdown(""" """)
         
    with col11:
        st.markdown("""**Visualize Data:** View data visualizations to see the distribution and trends of various symptoms and diseases. This helps in understanding the common health issues and their impact.""")
        st.markdown("""
                    - Our Visual Analytics feature enables you to view and interpret data on the distribution and trends of various symptoms and diseases.
                    - With interactive charts, graphs, and maps, you can explore data patterns and trends over time, across different demographics. This feature is especially useful for healthcare professionals, researchers, and policy makers looking to gain insights into public health trends.
                    - By visualizing data, you can better understand the relationships between symptoms and diseases, identify emerging health threats, and track the effectiveness of public health interventions.
                    - This powerful tool supports evidence-based decision-making and contributes to improving overall health outcomes.""")
        st.markdown(""" """)
    with col12:
        st.image("Images/Data Analysis.gif", use_column_width=True)
        
    with col13:
        st.image("Images/Online Health.gif", use_column_width=True)
    with col14:
        st.markdown("""**Self-Diagnosis:** Empower yourself with our advanced self-diagnosis tool. Simply input the symptoms you are experiencing, and our system will analyze them to predict possible illnesses you may be suffering from. This feature leverages a highly accurate Machine Learning model that boasts a 93% accuracy rate, ensuring reliable and insightful predictions.""")
        st.markdown("""This self-diagnosis feature is a significant advancement in personal healthcare management, providing users with a powerful tool to better understand their symptoms and potential health issues. With the integration of a highly accurate Machine Learning model, it offers reliable predictions and comprehensive health information to support your well-being. """)
        st.markdown("""**BENEFITS:**""")
        st.markdown("""
                    - Convenient and Accessible
                    - Educational
                    - Support for Healthcare Decisions""")
        st.markdown(""" """)
    
    symptoms_data = {
        "A": ["Asthma - is an inflammatory disorder of the airways, characterized by periodic attacks of wheezing, shortness of breath, chest tightness, and coughing.", "Abdominal pain - it may be felt anywhere between the bottom of your rib cage and your groin", "Acne - is a skin condition that is common in adolescents.", "Abscess - is an area under the skin where pus (infected fluid) collects.", "Allergies - are an immune system reaction to a substance called an allergen.","Anemia - is a low number of red blood cells or a low amount of hemoglobin in your red blood cells.", "Anxiety - is a condition that causes you to feel extremely worried or nervous.", "Arthritis - is pain or disease in one or more joints."],
        "B": ["Bronchitis - is a type of infection that affects your lungs.", "Bacterial Infection - Some bacteria cause disease in man, requiring treatment with an antibiotic.", "Blister - is a fluid-filled pocket on the surface of your skin.", "Botulism - is a rare but serious illness that attacks your nerves. Toxins (poison) from bacteria called Clostridium botulinum get into your bloodstream and attack your nerves.", "Bursitis - is inflammation of a bursa, a membrane-lined sac near a joint that acts as a cushion between the muscle and bone.", "Blurred Vision - is when you cannot see fine details.", "Bipolar Disorder - is a long-term chemical imbalance that causes rapid changes in mood and behavior.", "Bladder Infection - Inflammation of the urinary bladder.", "Bunion - is a bony lump at the base of your big toe. "],
        "C": ["Chronic Cough - is a cough that lasts more than 4 weeks in children or 8 weeks in adults.", "Cellulitis - is a skin infection caused by bacteria.", "Chronic Pain - is pain that does not get better for 3 months or longer.", "Cancer - is not a single disease. It's a group of diseases characterized by their ability to cause cells to change abnormally and grow out of control.", "COPD (Chronic Obstructive Pulmonary Disease) - is a long-term respiratory condition that often requires several different medications to control it, such as bronchodilators (short-acting or long-acting), corticosteroids, mucolytics, or antibiotics.", "Cirrhosis - is a disease in which normal liver cells are replaced by scar tissue, which interferes with all of these important functions. ", "Cold Symptoms - A cold is an infection caused by a virus. The infection causes your upper respiratory system to become inflamed. Common symptoms of a cold include sneezing, dry throat, a stuffy nose, headache, watery eyes, and a cough.", "Common Cold -  also called viral rhinitis, is one of the most common infectious diseases in humans.", "Constipation - means you have hard, dry bowel movements, or you go longer than usual between bowel movements.", "Concussion - is a mild brain injury.", "Chronic Back Pain - is back pain that lasts 3 months or longer."],
        "D": ["Diabetes Type1 - is a lifelong disorder caused by an autoimmune attack on the insulin-producing cells of the pancreas, which means the pancreas produces little or no insulin.", "Diabetes Type2 - Noninsulin-dependent Diabetes", "Dementia - is a condition that causes loss of memory, thought control, and judgment.", "Depression - is a mood disorder that causes feelings of sadness or hopelessness that do not go away.", "Diarrhea - is more frequent and more liquid bowel movements than normal.", "Dermatitis - is skin inflammation.", "Dermal Ulcer - is a skin sore with breakdown of tissue which may lead to a loss of epidermis, dermis or subcutaneous fat."],
        "E": ["Eczema - is an itchy, red, scaly skin rash.", "Edema - is swelling throughout your body, a sign that you are retaining fluid", "Eating Disorders - it include extreme emotions, attitudes, and behaviors surrounding weight and food issues.", "Elbow Sprain - is caused by a stretched or torn ligament in the elbow joint", "Emphysema - is a long-term lung disease", "Encephalitis - is inflammation of the brain", "Epilepsy - is a brain disorder that causes seizures", "Esophagitis - is inflammation or irritation of the lining of the esophagus"],
        "F": ["Flu - is a common infectious viral illness.", "Fever - normal body temperature is approximately 37°C. A fever is usually when your body temperature is 37.8°C or higher. You may feel warm, cold or shivery.", "Fibroids - are non-cancerous growths that develop in the muscular wall of the womb (uterus).", "Food Allergy - is when the body's immune system reacts unusually to specific foods. Although allergic reactions are often mild, they can be very serious.", "Functional neurological disorder (FND) - describes a problem with how the brain receives and sends information to the rest of the body."],
        "G": ["Gastritis - is inflammation or irritation of the lining of your stomach.", "Gastric Ulcer - are small holes or erosions that occur in the lining of your stomach.", "Gangrene - is a condition that happens when tissue dies.", "Galactosemia - is the inability of the body to use (metabolize) the simple sugar galactose (causing the accumulation of galactose 1-phosphate), which then reaches high levels in the body, causing damage to the liver, central nervous system, and various other body systems.", "GERD - Gastroesophageal reflux disease (GERD) is when food or liquid travels from the stomach back up into the esophagus.", "Goiter - is an enlargement of the thyroid gland.", "Glaucoma - is an eye disease that causes vision loss in one or both eyes"],
        "H": ["Hypertension - There needs to be a certain level of pressure in the arteries to move blood around the body. But, if blood pressure is higher than recommended over time it increases the risk of cardiovascular diseases like stroke or heart attack.", "HIV (human immunodeficiency virus) - The virus targets the immune system and if untreated, weakens your ability to fight infections and disease.", "Hiatus hernia - is when part of the stomach squeezes up into the chest through an opening (‘hiatus’) in the diaphragm.", "Heart Disease", "Hepatitis A - is a liver infection that’s spread in the poo of an infected person.", "Hepatitis B - is a liver infection that’s spread through blood and body fluids. ", "Headaches - Most headaches are not serious."],
        "I": ["Influenza", "Insomnia - can be both struggling to get to sleep or difficulty staying asleep", "Iron deficiency anaemia - is a condition where a lack of iron in the body leads to a reduction in the number of red blood cells.", "Itching - is an unpleasant sensation that compels a person to scratch the affected area.", "Indigestion - can be pain or discomfort in your upper abdomen (dyspepsia) or burning pain behind the breastbone (heartburn)."],
        "J": ["Jaundice"],
        "K": ["Kidney Stones - can develop in one or both kidneys and most often affect people aged 30 to 60.", "Kaposi’s sarcoma - is a rare type of cancer caused by a virus. It can affect the skin and internal organs.", "Kidney cancer - can include blood in your urine, a constant pain in your side, just below the ribs, a lump or swelling in the kidney area (on either side of the body)"],
        "L": ["Low blood pressure – sometimes referred to as hypotension, is a condition where the arterial blood pressure is abnormally low. Blood pressure is a measure of the force that your heart uses to pump blood around your body. ", "Lupus - is a complex and poorly understood condition that affects many parts of the body and causes symptoms ranging from mild to life-threatening.", "Labyrinthitis - is an inner ear infection. It causes the labyrinth inside your ear to become inflamed, affecting your hearing and balance.", "Lactose intolerance - is a common digestive problem where the body is unable to digest lactose, a type of sugar mainly found in milk and dairy products.", "Leg cramps - are a common and usually harmless condition. They cause the muscles in your leg to suddenly become tight and painful.", "Lymphoedema - is a chronic (long-term) condition that causes swelling in the body’s tissues. It can affect any part of the body, but usually develops in the arms or legs."],
        "M": ["Migraine - is a common health condition. It affects around 1 in every 5 women and around 1 in every 15 men. It usually begins in early adulthood.", "Malnutrition - means poor nutrition. Most commonly this is caused by not eating enough (undernutrition) or not eating enough of the right food to give your body the nutrients it needs.", "Malignant brain tumour - is a fast-growing cancer that spreads to other areas of the brain and spine.", "Mouth ulcers - are painful sores that appear in the mouth. They’re uncomfortable but they’re usually harmless."],
        "N": ["Narcolepsy"],
        "O": ["Osteoporosis"],
        "P": ["Pneumonia"],
        "Q": ["Q Fever"],
        "R": ["Rheumatoid Arthritis"],
        "S": ["Sinusitis"],
        "T": ["Tuberculosis"],
        "U": ["Ulcer"],
        "V": ["Vertigo"],
        "W": ["Whooping Cough"],
        "X": ["Xerosis"],
        "Y": ["Yellow Fever"],
        "Z": ["Zika Virus"],
    }
    st.markdown( """
        <style>
        .button {
            display: inline-block;
            border-radius: 80%;
            background-color: #f0f0f0;
            border: none;
            color: black;
            text-align: center;
            font-size: 30px;
            padding: 20px;
            width: 60px;
            height: 60px;
            transition: all 0.3s;
            cursor: pointer;
            margin: 5px;
        }
        .button:hover {
            background-color: #e0e0e0;
        }
        .container {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    def display_symptoms(symptom_list):
        for symptom in symptom_list:
            st.markdown(symptom)

    def main():
        st.markdown("## Search diseases & conditions")
        st.write("You will find a currated list of symptoms and diseases in alphabetical order below:")

        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        symptoms_selected = False

        rows = 6
        cols = 3
        index = 0

        for row in range(rows):
            cols = st.columns(5)
            for col in cols:
                if index < len(alphabet):
                    letter = alphabet[index]
                    if col.button(letter):
                        symptoms_selected = True
                        if letter in symptoms_data:
                            st.write(f"**Symptoms starting with '{letter}':**")
                            display_symptoms(symptoms_data[letter])
                        else:
                            st.write(f"No symptoms found for '{letter}'")
                    index += 1

        if not symptoms_selected:
            st.write("Click on a letter to see symptoms starting with that letter.")

    if __name__ == "__main__":
        main()


if selected == "Understanding our Data":
    df1 = pd.read_csv("dataset.csv")
    df2 = pd.read_csv("Symptom-severity.csv")
    
    st.markdown("#### Data visualisation in Symptoms and disease Dataset")
    st.write(df1.head(10))
    st.markdown("#### Symptom Severity Dataset")
    st.write(df2)
    
    df_cleaned = clean_data(df1, df2)
    # Visualize the data
    fig1, fig2, fig3, fig4, fig5 = visualize_data(df_cleaned)
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)
    st.plotly_chart(fig4)
    st.plotly_chart(fig5)
 
if selected == "Self-Diagnosis":
    # Symptoms List
    all_symptoms = ['skin_rash', 'itching', 'nodal_skin_eruptions', 'dischromic_patches',
                        'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 
                        'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 
                        'spotting_urination', 'fatigue', 'weight_gain']
    st.markdown("#### Select the symptoms you are facing ...")
    num_columns = 4
    columns = st.columns(num_columns)
    
    symptoms = []
    for i, symptom in enumerate(X_columns):
        col = columns[i % num_columns]
        if symptom in symptom_encoders:
            selected_symptom = col.selectbox(f"{symptom.capitalize().replace('_', ' ')}", [""] + all_symptoms)
            symptoms.append(selected_symptom if selected_symptom else 0)
        else:
            symptoms.append(0)
                
    if st.button("Predict Disease"):
        predicted_disease = predict_disease(symptoms)
        st.markdown("*According to the prediction, you maybe suffering from:*")
        st.success(f' {predicted_disease}')






