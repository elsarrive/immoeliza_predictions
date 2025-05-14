import XGBoost_model as m
import cleaning_dataset as cl
import streamlit as sl
import pandas as pd
import pickle

##############################################
#### IMPORTATION MODEL AND INITIALIZATION ####
##############################################

@sl.cache_resource  
def load_model():
    with open("model_and_scaler/scaler.pkl", "rb") as f_scaler:
        scaler = pickle.load(f_scaler)
    with open("model_and_scaler/model.pkl", "rb") as f_model:
        model = pickle.load(f_model)
    return scaler, model

scaler, model = load_model()

HasAttic = False  
HasBasement = False  
HasDressingRoom = False  
HasDinningRoom = False  
has_lift = False  
HasHeatPump = False  
HasPhotovoltaicPanels = False  
HasThermicPanels = False  
HasLivingRoom = False  
has_garden = False  
HasAirConditioning = False  
HasArmoredDoor = False  
HasVisiophone = False  
HasOffice = False  
has_swimming = False  
HasFireplace = False  
has_terrace = False  

def user_input_preparation(input_dict):
    input_dict['id'] = 0
    input_dict['url'] = ""
    input_dict['roomCount'] = 0
    input_dict['monthlyCost'] = 0
    input_dict['diningRoomSurface'] = 0
    input_dict['terraceOrientation'] = ""
    input_dict['accessibleDisabledPeople'] = False
    input_dict['floorCount'] = 0
    input_dict['streetFacadeWidth'] = 0
    input_dict['kitchenSurface'] = 0
    input_dict['livingRoomSurface'] = 0
    input_dict['hasBalcony'] = False
    input_dict['gardenOrientation'] = ""
    input_dict['price'] = 0

    return input_dict


##############################################
###### START OF THE STREAMLIT INTERFACE ######
##############################################

def markdown20(text):
    sl.markdown(f"<span style='font-size:20px'>{text}</span>", unsafe_allow_html=True)

sl.title("ImmoEliza")
sl.header("🏠 Real Estate Price Prediction 🏠")

# GENERAL INFORMATIONS
sl.markdown("----")
sl.subheader('ℹ️ General informations')
Type = sl.selectbox("House or appartment?", ["HOUSE", "APPARTMENT"])

if Type == 'APPARTMENT': 
    has_lift = sl.checkbox('Has a lift')
    Subtype = sl.selectbox("Which subtype?", ['APARTMENT', 'FLAT_STUDIO', 'DUPLEX', 'PENTHOUSE','APARTMENT_GROUP', 'GROUND_FLOOR', 'APARTMENT_BLOCK', 
    'MIXED_USE_BUILDING', 'TRIPLEX', 'LOFT','CHALET', 'SERVICE_FLAT', 'KOT', 'BUNGALOW','OTHER_PROPERTY'])

if Type == 'HOUSE':
    Subtype = sl.selectbox("Which subtype?", ["HOUSE", "TOWN_HOUSE", "VILLA", "CHALET", "BUNGALOW",
    "COUNTRY_COTTAGE", "MANOR_HOUSE", "MANSION", "EXCEPTIONAL_PROPERTY", "CASTLE", "FARMHOUSE", "HOUSE_GROUP", "OTHER_PROPERTY", "PAVILION"])

BuildingCondition = sl.selectbox("Which condition?", cl.df.buildingCondition.dropna().unique())
BuildingConstructionYear = sl.number_input(label="Construction year", min_value=1850, max_value=2050, value= 2000)
FacadeCount = sl.number_input(label= "Number of facades", min_value=1, max_value=20)

# GEOGRAPHICAL INFORMATIONS
sl.markdown("----")
sl.subheader("📍 Geographical informations")
Province = sl.selectbox("Province", ['Brussels', 'Luxembourg', 'Antwerp', 'Flemish Brabant',
       'East Flanders', 'West Flanders', 'Liège', 'Walloon Brabant',
       'Limburg', 'Namur', 'Hainaut'])
Locality = sl.selectbox("Locality", sorted(cl.df.locality.unique()))
Postcode = sl.selectbox("Postcode", sorted(cl.df.postCode.unique()))

# ROOMS
sl.markdown("----")
sl.subheader('🛏️ Rooms')
markdown20('How many bedrooms in your property?')
BedroomCount = sl.slider("Bedrooms", 1, 8, 2)
markdown20('How many bathrooms in your property?')
BathroomCount = sl.slider("Bathrooms", 1, 5, 2)
markdown20('How many toilets in your property?')
ToiletCount = sl.slider("Toilets", 1, 6, 2)
HasDinningRoom = sl.checkbox('Has a dinning room')
HasLivingRoom = sl.checkbox('Has a living room')
HasDressingRoom = sl.checkbox('Has a dressing room')
HasOffice = sl.checkbox('Has an office')
HasAttic = sl.checkbox('Has an attic')
HasBasement = sl.checkbox('Has a basement')
KitchenType = sl.selectbox('What\'s the kitchen type?', ['NOT_INSTALLED', 'SEMI_EQUIPPED', 'INSTALLED', 'HYPER_EQUIPPED'])

# SURFACES
sl.markdown("----")
sl.subheader('📐 Surfaces')
markdown20('What\'s the size of your property?')
HabitableSurface = sl.slider("Habitable surface", 0, 800, 200, step=25)
markdown20('What\'s the land surface?')
LandSurface = sl.slider("Land surface", 0, 2000, 150, step=25)

# EXTERIOR SPACE
sl.markdown("----")
sl.subheader('🌳 Exterior space')
has_terrace = sl.checkbox("Has a terrace")
if has_terrace:
    terrace_area = sl.slider("What's the size of the terrace?", 0, 250, 15, step=5)
    sl.write("Superficie sélectionnée :", terrace_area)
else:
    terrace_area = None

has_garden = sl.checkbox("Has a garden")
if has_garden:
    garden_area = sl.slider("What's the size of the garden?", 0, 2000, 150, step=25)
    sl.write("Superficie sélectionnée :", garden_area)
else:
    garden_area = None

has_swimming = sl.checkbox('Has a swimmingpool')
flood_zone = sl.selectbox("There is a kind of flood zone?", cl.df.floodZoneType.dropna().unique())

# ENERGY
sl.markdown("----")
sl.subheader('♻️ Energy')
HasHeatPump = sl.checkbox('Has heat pump')
HasPhotovoltaicPanels = sl.checkbox('Has photovoltaic panels')
HasThermicPanels = sl.checkbox('Has thermic panels')
HasAirConditioning = sl.checkbox('Has air conditionning')
EpcScore = sl.selectbox('What\'s the EPC (PEB) score?', ["A++", "A+", "A", "B", "C", "D", "E", "F", "G"])
HeatingType = sl.selectbox('What\'s the heating type?', sorted(cl.df.heatingType.dropna().unique()))

# EXTRA
sl.markdown("----")
sl.subheader('✨ Extra')
HasFireplace = sl.checkbox('Has a fire place')
HasArmoredDoor = sl.checkbox('Has an armored door')
HasVisiophone = sl.checkbox('Has a visiophone')
ParkingIndoor = sl.slider(label= 'Parking places indoor', min_value=0, max_value=5)
ParkingOutdoor = sl.slider(label= 'Parking places outdoor', min_value=0, max_value=3)

# PREDICTION
sl.markdown("----")
if sl.button('Prediction'):
    input_dict = { # LINK TO THE MODEL'S COLUMNS NAMES
        'type' : Type, 
        'subtype' : Subtype,
        'bedroomCount' : BedroomCount, 
        'bathroomCount' : BathroomCount,
        'province' : Province,
        'locality' : Locality,
        'postCode' : Postcode,
        'habitableSurface' : HabitableSurface,
        'hasAttic' : HasAttic,
        'hasBasement' : HasBasement,
        'hasDressingRoom' : HasDressingRoom,
        'hasDiningRoom' : HasDinningRoom,
        'buildingCondition' : BuildingCondition,
        'buildingConstructionYear' : BuildingConstructionYear,
        'facedeCount' : FacadeCount,
        'hasLift' : has_lift,
        'floodZoneType' : flood_zone,
        'heatingType' : HeatingType,
        'hasHeatPump' : HasHeatPump,
        'hasPhotovoltaicPanels' : HasPhotovoltaicPanels,
        'hasThermicPanels' : HasThermicPanels,
        'kitchenType' : KitchenType,
        'landSurface' : LandSurface,
        'hasLivingRoom' : HasLivingRoom,
        'hasGarden' : has_garden,
        'gardenSurface' : garden_area,
        'parkingCountIndoor' : ParkingIndoor,
        'parkingCountOutdoor' : ParkingOutdoor,
        'hasAirConditioning' : HasAirConditioning,
        'hasArmoredDoor' : HasArmoredDoor,
        'hasVisiophone' : HasVisiophone,
        'hasOffice' : HasOffice,
        'toiletCount' : ToiletCount,
        'hasSwimmingPool' : has_swimming,
        'hasFireplace' : HasFireplace,
        'hasTerrace' : has_terrace,
        'terraceSurface' : terrace_area,
        'epcScore' : EpcScore
        }
    
    # add features that is not required by user input (but required by the functions associated with the model)
    input_dict_complete = user_input_preparation(input_dict)

    # creation of a dataframe with the input dictionnary 
    df_new = pd.DataFrame([input_dict_complete])

    # cleaning function
    df_new_clean = cl.cleaning_dataframe(df_new, is_training=False)

    # imputations
    df_new_clean_imputed = cl.transform_cleaning_traintestsplit(df_new_clean, m.stats_from_X_train, is_training=False)

    # dataframe with the columns in the correct order  -> scaling
    columns_for_model = scaler.feature_names_in_.tolist()
    for col in columns_for_model:
        if col not in df_new_clean_imputed:
            df_new_clean_imputed[col] = -1 # to be sure not have missing values

    df_new_clean_imputed = df_new_clean_imputed[columns_for_model]
    df_new_scaled = scaler.transform(df_new_clean_imputed)

    # prediction
    predicted_price = model.predict(df_new_scaled)

    sl.write(f"Le prix prédit est {predicted_price[0]:,.0f} €".replace(',', ' '))
