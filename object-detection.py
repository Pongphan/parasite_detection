import streamlit as st
from PIL import Image
import os

#path = "E:/P Works/My Projects/Project - ทุน ศ สูง มข/web_component/"
path = ""  #use this path for deploy

#--------------------------------------------------------------------------------------------------
#login_page
#--------------------------------------------------------------------------------------------------
import csv
from datetime import datetime

def login_page():
    CSV_FILE = path + "userlogin.csv"

    st.title("User Information")

    with st.form(key='user_form'):
        name = st.text_input("Name")
        surname = st.text_input("Surname")
        affiliation = st.text_input("Affiliation")
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([name, surname, affiliation, timestamp])
        st.success("Thank you!")
        st.session_state.logged_in = True
#--------------------------------------------------------------------------------------------------
#home_page
#--------------------------------------------------------------------------------------------------
def home_page():
    st.title("A platform for pinworm detection")
    st.subheader("Unlock expert pinworm detection knowledge at your fingertips — completely free for everyone.")
    st.write("A platform for pinworm detection was developed by the High Potential Research Team Grant Program (Contract no. N42A670561 to Wanchai Maleewong (WM)).")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Parasited image")
        
        #------------------------------------------------------------------------------------------
        img_1 = Image.open(path + "ev-web/input_3141.tif")
        img_2 = Image.open(path + "ev-web/output_3141.png")
        #------------------------------------------------------------------------------------------

        st.image(img_1, caption="Input image", use_container_width=True)
        st.image(img_2, caption="Detection image", use_container_width=True)

    with col2:
        st.subheader("Parasited image")

        #------------------------------------------------------------------------------------------
        img_3 = Image.open(path + "ev-web/input_3146.tif")
        img_4 = Image.open(path + "ev-web/output_3146.png")
        #------------------------------------------------------------------------------------------

        st.image(img_3, caption="Input image", use_container_width=True)
        st.image(img_4, caption="Detection image", use_container_width=True)
#--------------------------------------------------------------------------------------------------
#pinworm_detail_page
#--------------------------------------------------------------------------------------------------
def pinworm_detail_page():

    st.sidebar.title("Table of Contents")
    st.sidebar.markdown("""
    - [Geographical Distribution](#geographical-distribution)
    - [Morphology](#morphology)
    - [Life Cycle](#life-cycle)
    - [Mode of Transmission](#mode-of-transmission)
    - [Route of Infection](#route-of-infection)
    - [Laboratory Diagnosis](#laboratory-diagnosis)
    - [Treatment](#treatment)
    - [Prevention and Control](#prevention-and-control)
    - [References](#references)
    """)

    st.title("Enterobius vermicularis (Pinworm) Comprehensive Guide")

    st.header("Geographical Distribution")
    st.write("""
    Enterobius vermicularis (pinworm) has a worldwide distribution with no geographical limitations. It is most prevalent in temperate regions and is considered the most common helminthic infection in developed countries, particularly in North America, Europe, parts of Asia, Australia, and New Zealand. The infection rates are highest in crowded environments such as schools, childcare facilities, and institutional settings. Infection rates can reach 50% in some pediatric populations, with estimates suggesting that hundreds of millions of people are infected globally at any given time.
    """)

    st.header("Morphology")
    st.subheader("Adult Worm")
    st.markdown("""
    - **Female:** 8-13 mm long, 0.5 mm wide, white-colored with a characteristic pointed posterior end forming a long, pin-like tail (hence the name "pinworm"). The anterior end has three distinctive lips surrounding the mouth and cervical alae (lateral expansions).
    - **Male:** Smaller than the female, measuring 2-5 mm long, with a curved posterior end and a prominent copulatory spicule.
    - **Both sexes:** Possess a bulbous esophagus with a posterior bulb containing a valve apparatus.
    """)
    st.subheader("Egg")
    st.markdown("""
    - **Shape:** Elongated oval, flattened on one side giving an asymmetrical appearance.
    - **Size:** 50-60 μm long, 20-30 μm wide.
    - **Color:** Transparent, ranging from colorless to slightly yellowish.
    - **Shell:** Thin but double-contoured.
    - **Content:** Contains a partially developed embryo (at various stages of cleavage) when laid.
    """)
    col1, col2, col3 = st.columns(3)

    #------------------------------------------------------------------------------------------
    img_1 = Image.open(path + "ev-web/egg_0016.png")
    img_2 = Image.open(path + "ev-web/egg_0020.png")
    img_3 = Image.open(path + "ev-web/egg_0023.png")
    #------------------------------------------------------------------------------------------

    with col1:
        st.image(img_1, use_container_width=True)
    with col2:
        st.image(img_2, use_container_width=True)
    with col3:
        st.image(img_3, use_container_width=True)

    st.header("Life Cycle")
    st.markdown("""
    1. **Egg deposition:** Gravid female worms migrate out of the anus at night and deposit eggs on the perianal skin.
    2. **Embryonation:** Eggs become infective within 4-6 hours after being laid.
    3. **Transmission:** Eggs are transferred to the mouth via contaminated hands, food, or fomites.
    4. **Hatching:** After ingestion, eggs hatch in the duodenum, releasing larvae.
    5. **Maturation:** Larvae migrate down the small intestine to the cecum and colon where they mature into adults.
    6. **Mating:** Male and female worms mate in the large intestine.
    7. **Migration:** Gravid females migrate to the perianal region to deposit eggs, typically at night.
    8. **Complete cycle:** The entire life cycle from ingestion to egg production takes approximately 2-6 weeks.

    *Note: An autoinfection cycle can occur when eggs hatch on the perianal skin and larvae migrate back into the anus, reaching the large intestine.*
    """)

    st.header("Mode of Transmission")
    st.markdown("""
    - **Direct contact:** Person-to-person transmission through contaminated hands.
    - **Fomite transmission:** Via contaminated clothing, bedding, toys, or other objects.
    - **Airborne transmission:** Eggs are lightweight and can become airborne in dust particles.
    - **Retroinfection:** Rarely, eggs may hatch on perianal skin and larvae migrate back into the anus.
    - **Autoinfection:** Hand-to-mouth transfer after scratching the perianal area.
    """)

    st.header("Route of Infection")
    st.markdown("""
    1. **Oral ingestion:** The primary route through contaminated hands, food, or objects.
    2. **Inhalation:** Eggs may be inhaled with dust and subsequently swallowed.
    3. **Retroinfection:** Direct reinfection from eggs hatching on the perianal skin (uncommon).

    The infectious dose is low; ingestion of only a few eggs can establish an infection.
    """)

    st.header("Laboratory Diagnosis")
    st.subheader("Specimen Collection")
    st.markdown("""
    - **Scotch tape Test:** The gold standard for detection.
    - Transparent adhesive tape applied to the perianal region in the early morning before bathing or defecation.
    - Optimal timing: 2-3 hours after the patient has gone to sleep.
    - Should be performed on 3 consecutive days to increase sensitivity (detection rates increase from 50% with one sample to 90% with three samples).
    - **Perianal Swab:** An alternative to the tape test.
    - Uses a moistened cotton swab applied to the perianal area.
    - Less sensitive than the tape test but easier to perform in certain settings.
    - **Stool Examination:** Not recommended as a primary diagnostic method.
    - Low sensitivity (≤5-15%) because eggs are laid on the perianal skin, not in stool.
    - Adult worms may occasionally be found in stool samples.
    """)
    st.subheader("Microscopic Examination")
    st.markdown("""
    - **Direct Examination:**
    - Tape or swab is placed on a glass slide with a drop of clearing agent (xylene or toluene).
    - Examined under low power (10x) and then under higher magnification (40x) to confirm.
    - Characteristic eggs appear colorless, asymmetrically oval, and flattened on one side.
    - **Special Techniques:**
    - NIH swab: Calcium alginate swab moistened with saline.
    - Anal impression: Using filter paper pressed against the perianal area.
    - Cellophane paddle devices: Commercial collection systems.
    """)
    st.subheader("Macroscopic Diagnosis")
    st.markdown("""
    - Visual identification of adult worms (white, thread-like, 5-13 mm) in the perianal region, stool, or underwear.
    - Night inspection of the perianal area with a flashlight may reveal migrating female worms.
    """)
    st.subheader("Molecular Methods")
    st.markdown("""
    - **PCR-based detection:** Performed on perianal swabs or tape tests.
    - Offers higher sensitivity than conventional microscopy.
    - Not routinely used in clinical practice due to cost and infrastructure requirements.
    """)
    st.subheader("Quality Control Considerations")
    st.markdown("""
    - Proper specimen storage (should be examined within 24 hours).
    - Adequate training of laboratory personnel in egg morphology.
    - Attention to timing (false negatives may occur if collected after bathing).
    - Multiple samples increase detection rates.
    """)

    st.header("Treatment")
    st.subheader("Pharmacological Treatment")
    st.markdown("""
    **Pyrantel Pamoate:**
    - Dosage: 11 mg/kg as a single dose (maximum 1 gram).
    - Mechanism: Causes neuromuscular blockade in the worm.
    - Available over-the-counter in many countries.
    - Safe for children ≥2 years and pregnant women.

    **Mebendazole:**
    - Dosage: 100 mg as a single dose for all ages.
    - Mechanism: Inhibits microtubule formation and glucose uptake in the worm.
    - Requires a prescription in most countries.
    - Contraindicated in pregnancy (first trimester).

    **Albendazole:**
    - Dosage: 400 mg as a single dose for adults and children >2 years.
    - Mechanism: Similar to mebendazole.
    - Offers a broader spectrum against other helminths.
    - Contraindicated in pregnancy (first trimester).
    """)
    st.subheader("Treatment Protocols")
    st.markdown("""
    - Single dose treatment followed by a second dose after 2 weeks.
    - Simultaneous treatment of all household members regardless of symptoms.
    - Treatment of close contacts in institutional settings.
    """)
    st.subheader("Symptomatic Treatment")
    st.markdown("""
    - Use of antipruritic creams or ointments for perianal itching.
    - Sitz baths to relieve irritation.
    - Good perianal hygiene to prevent secondary bacterial infections.
    """)
    st.subheader("Treatment Challenges")
    st.markdown("""
    - High reinfection rates (30-90%).
    - Multiple treatment courses may be necessary.
    - Resistance to anthelmintics is rare but has been reported.
    """)
    st.subheader("Special Populations")
    st.markdown("""
    - **Pregnant Women:** Pyrantel pamoate is preferred; mebendazole and albendazole are contraindicated in the first trimester.
    - **Children <2 Years:** Dosage adjustments are required; treatment should be supervised by pediatricians.
    - **Immunocompromised Patients:** May require longer treatment courses.
    """)
    st.subheader("Follow-up")
    st.markdown("""
    - Repeat the tape test 2-3 weeks after treatment completion.
    - Additional treatment cycles if reinfection occurs.
    - Emphasize preventive measures to break the transmission cycle.
    """)

    st.header("Prevention and Control")
    st.subheader("Personal Hygiene Measures")
    st.markdown("""
    - Thorough handwashing with soap and water, especially after using the toilet and before eating.
    - Regular showering or bathing—particularly in the morning to remove eggs.
    - Keeping fingernails short and clean.
    - Avoiding nail-biting and finger-sucking behaviors.
    """)
    st.subheader("Environmental Management")
    st.markdown("""
    - Daily changing and laundering of underwear, pajamas, and bed linens in hot water.
    - Regular vacuuming of living spaces and cleaning of surfaces.
    - Careful handling and washing of potentially contaminated toys and objects.
    - Reducing bedroom dust by simplifying furnishings.
    """)
    st.subheader("Medical Interventions")
    st.markdown("""
    - Prompt treatment of infected individuals with anthelmintic medications (pyrantel pamoate, mebendazole, or albendazole).
    - Simultaneous treatment of all household members, even if asymptomatic.
    - Repeated treatment 2 weeks after the initial dose to eliminate newly hatched worms.
    - Follow-up testing to confirm clearance of infection.
    """)
    st.subheader("Institutional Control")
    st.markdown("""
    - Education of childcare workers, teachers, and parents about transmission and prevention.
    - Implementation of hygiene protocols in schools and childcare facilities.
    - Proper sanitization of communal spaces and bathrooms.
    - Regular health screening in institutional settings.
    """)

    st.header("References")
    st.markdown("""
    1. **Arora, D. R., & Arora, B. (2014).** *Medical Parasitology* (4th ed.). CBS Publishers & Distributors.  
    2. **Centers for Disease Control and Prevention. (2019).** *Enterobiasis (Pinworm Infection).* [CDC](https://www.cdc.gov/parasites/pinworm/index.html)  
    3. **Cook, G. C., & Zumla, A. I. (2023).** *Manson's Tropical Diseases* (24th ed.). Elsevier Health Sciences.  
    4. **Garcia, L. S. (2016).** *Diagnostic Medical Parasitology* (6th ed.). ASM Press.  
    5. **Kucik, C. J., Martin, G. L., & Sortor, B. V. (2004).** Common intestinal parasites. *American Family Physician, 69*(5), 1161-1168.  
    6. **Lohiya, G. S., Tan-Figueroa, L., Crinella, F. M., & Lohiya, S. (2000).** Epidemiology and control of enterobiasis in a developmental center. *Western Journal of Medicine, 172*(5), 305-308.  
    7. **World Health Organization. (2020).** *Helminth Control in School-Age Children: A Guide for Managers of Control Programmes* (2nd ed.). WHO Press.
    """)
#--------------------------------------------------------------------------------------------------
#quiz_page
#--------------------------------------------------------------------------------------------------
def quiz_page():
    st.header("Parasitic Examination")
    st.write("Test your knowledge with our interactive examinations.")

    with st.form(key="quiz_form"):
        q1 = st.radio(
            "Question 1: Which characteristic is most diagnostic of Enterobius vermicularis eggs?",
            ["Round shape with thick shell","Flattened on one side, with asymmetrical appearance","Contains a fully developed larva","Brown coloration","Presence of polar plugs at both ends"]
        )
        q2 = st.radio(
            "Question 2: What is the optimal time to collect samples for E. vermicularis detection?",
            ["Early morning before defecation","After antiparasitic medication","During symptomatic episodes","Midday with fecal sample","Before bedtime"]
        )
        q3 = st.radio(
            "Question 3: Which microscopic feature helps distinguish E. vermicularis adult females from other intestinal nematodes?",
            ["Presence of lateral alae","Pointed posterior end","Bulbous esophagus","Presence of cervical wings","Coiled appearance"]
        )
        q4 = st.radio(
            "Question 4: What is the primary location where E. vermicularis eggs are typically found during examination?",
            ["In concentrated stool samples","Perianal skin folds","Within the intestinal mucosa","In urine sediment","Blood smears"]
        )
        q5 = st.radio(
            "Question 5: Which staining technique is most appropriate for visualizing E. vermicularis eggs?",
            ["Gram stain","Acid-fast stain","No stain needed (best examined unstained)","Giemsa stain","Trichrome stain"]
        )
        submit_button = st.form_submit_button(label="Submit Answers")

    if submit_button:
        score = 0
        if q1 == "Flattened on one side, with asymmetrical appearance":
            score += 1
        if q2 == "Early morning before defecation":
            score += 1
        if q3 == "Pointed posterior end":
            score += 1
        if q4 == "Perianal skin folds":
            score += 1
        if q5 == "No stain needed (best examined unstained)":
            score += 1

        st.success(f"Your score is {score} from 5")
#--------------------------------------------------------------------------------------------------
#datadet (a component in ai_detector_page)
#--------------------------------------------------------------------------------------------------
def dataset():
    st.subheader("A Sample Dataset")
    col1,col2,col3,col4 = st.columns(4)

    img_1 = Image.open(path + "sample_objdet/evegg (3000).tif")
    img_2 = Image.open(path + "sample_objdet/evegg (3001).tif")
    img_3 = Image.open(path + "sample_objdet/evegg (3002).tif")
    img_4 = Image.open(path + "sample_objdet/evegg (3003).tif")
    img_5 = Image.open(path + "sample_objdet/evegg (3004).tif")

    img_6 = Image.open(path + "sample_objdet/evegg (3006).tif")
    img_7 = Image.open(path + "sample_objdet/evegg (3007).tif")
    img_8 = Image.open(path + "sample_objdet/evegg (3010).tif")
    img_9 = Image.open(path + "sample_objdet/evegg (3011).tif")
    img_10 = Image.open(path + "sample_objdet/evegg (3012).tif")

    img_11 = Image.open(path + "sample_objdet/evegg (3014).tif")
    img_12 = Image.open(path + "sample_objdet/evegg (3015).tif")
    img_13 = Image.open(path + "sample_objdet/evegg (3016).tif")
    img_14 = Image.open(path + "sample_objdet/evegg (3020).tif")
    img_15 = Image.open(path + "sample_objdet/evegg (3023).tif")

    img_16 = Image.open(path + "sample_objdet/evegg (3024).tif")
    img_17 = Image.open(path + "sample_objdet/evegg (3025).tif")
    img_18 = Image.open(path + "sample_objdet/evegg (3028).tif")
    img_19 = Image.open(path + "sample_objdet/evegg (3030).tif")
    img_20 = Image.open(path + "sample_objdet/evegg (3037).tif")

    with col1:
        st.image(img_1)
        st.image(img_2)
        st.image(img_3)
        st.image(img_4)
        st.image(img_5)
    with col2:
        st.image(img_6)
        st.image(img_7)
        st.image(img_8)
        st.image(img_9)
        st.image(img_10)
    with col3:
        st.image(img_11)
        st.image(img_12)
        st.image(img_13)
        st.image(img_14)
        st.image(img_15)
    with col4:
        st.image(img_16)
        st.image(img_17)
        st.image(img_18)
        st.image(img_19)
        st.image(img_20)

    st.write("Select the images you wish to download:")

#--------------------------------------------------------------------------------------------------
#ai_detector_page
#--------------------------------------------------------------------------------------------------
import cv2
import numpy as np
import tensorflow as tf

def ai_detector_page():
    st.title("AI Detector")

    with st.expander("Sample Dataset"):
        dataset()

    option = st.selectbox(
    "Selected AI Model",
    ("BX43+DP27_cnn_ev", "BX43+DP27_cnn_ev_v2", "BX43+DP27_v3"),)
    if option != "BX43+DP27_cnn_ev":
        st.write("**Caution** This model is currently not available. Model **BX43+DP27_cnn_ev** will be used instead.")

    st.subheader("Upload & View Image")
    st.write("Upload an image and view it below.")

    #----------------------------------------------------------------------------------------------
    model_path = path + "model/aug_img_cnn.h5"
    #----------------------------------------------------------------------------------------------

    model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

    def boxlocation(img_c, box_size):
        non_zero_points = np.argwhere(img_c > 0)
        if non_zero_points.size == 0:
            return None

        y_min, x_min = np.min(non_zero_points, axis=0)
        y_max, x_max = np.max(non_zero_points, axis=0)

        return [y_min - box_size, y_max + box_size, x_min - box_size, x_max + box_size]

    def drawbox(img, label, a, b, c, d, box_size):
        image = cv2.rectangle(img, (c, a), (d, b), (0, 255, 0), 2)
        image = cv2.putText(image, label, (c + box_size, a - 10), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 255), 1)
        return image

    def objectdet(img):
        img = cv2.resize(img, (img.shape[1] // 1, img.shape[0] // 1), interpolation=cv2.INTER_AREA)
        
        box_size_y, box_size_x = 370, 370
        step_size = 50
        img_output = np.array(img)
        img_cont = np.zeros((img_output.shape[0], img_output.shape[1]), dtype=np.uint8)
        result = 0

        for i in range(0, img_output.shape[0] - box_size_y, step_size):
            for j in range(0, img_output.shape[1] - box_size_x, step_size):
                img_patch = img_output[i:i + box_size_y, j:j + box_size_x]
                img_patch = cv2.resize(img_patch, (128, 128), interpolation=cv2.INTER_AREA)
                img_patch = np.expand_dims(img_patch, axis=0)

                y_outp = model.predict(img_patch, verbose=0)

                if result < y_outp[0][1] and y_outp[0][1] > 0.95:
                    result = y_outp[0][1]
                    img_cont[i + (box_size_y // 2), j + (box_size_x // 2)] = int(y_outp[0][1] * 255)

        if result != 0:
            label = f"ev: {result:.2f}"
            boxlocat = boxlocation(img_cont, box_size_x // 2)
            if boxlocat:
                img_output = drawbox(img, label, *boxlocat, box_size_x // 2)

        return img_output

    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "tif"])
    if uploaded_file is not None:
        try:
            image = np.array(Image.open(uploaded_file))
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            st.image(image, caption="Uploaded Image")

            output_img = objectdet(image)
            st.image(output_img, caption="Processed Image")

        except Exception as e:
            st.error(f"Error loading image: {e}")
#--------------------------------------------------------------------------------------------------
#about_page
#--------------------------------------------------------------------------------------------------
def about_page():
    st.title("Funding")
    st.markdown("---")

    #----------------------------------------------------------------------------------------------
    image = Image.open(path + "page/banner.jpg")
    #----------------------------------------------------------------------------------------------
    
    st.image(image, use_container_width=True)
    
    st.markdown("---")
    st.subheader("This project was funded by the grant from the National Research Council of Thailand (NRCT):")
    st.write("""High-Potential Research Team Grant Program (Contract no. N42A670561 to Wanchai Maleewong (WM)). The contents of this report are solely the responsibility of the authors and do not necessarily represent the official views of the NRCT.""")

    with st.expander("Contact Information"):
        st.write("Contact system administrator via E-mail:")
        st.write("- **Email:** mufhasa8165@hotmail.com")
    
    st.markdown("---")
    st.write("**Platform version 1**: release 25 February 2025")
    st.write("© 2567 Research Group | All rights reserved")
#--------------------------------------------------------------------------------------------------
# #main function
#--------------------------------------------------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
else:

    #----------------------------------------------------------------------------------------------
    image = Image.open(path + "page/Monogram-Logo-02-320x160.png")
    #----------------------------------------------------------------------------------------------

    st.sidebar.image(image, use_container_width=True)
    st.sidebar.markdown("<div class='sidebar-title'>Platform Navigation</div>", unsafe_allow_html=True)
    
    pages = {
        "Home": home_page,
        "Pinworm Detail": pinworm_detail_page,
        "Quiz": quiz_page,
        "AI Detector": ai_detector_page,
        "About": about_page,
    }
    
    selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))
    pages[selected_page]()
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()
