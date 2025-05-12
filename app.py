import streamlit as sl
import requests
import predictions_elsa as p
import numpy as np 

Type = ""  # string, obligatoire pour Pydantic: valeur vide = missing
Subtype = ""  # string
BedroomCount = 0  # nombre de chambres (int ou float, selon backend)
BathroomCount = 0  # int/float
Province = ""  # string
Locality = ""  # string
Postcode = ""  # string (ou int si c'est géré ainsi côté backend)
HabitableSurface = 0  # int/float surface
HasAttic = False  # bool
HasBasement = False  # bool
HasDressingRoom = False  # bool
HasDinningRoom = False  # bool
BuildingCondition = ""  # string
BuildingConstructionYear = None  # int/float ou None si inconnu
FacadeCount = 0  # int
has_lift = False  # bool
flood_zone = None  # string ou None
HeatingType = ""  # string
HasHeatPump = False  # bool
HasPhotovoltaicPanels = False  # bool
HasThermicPanels = False  # bool
KitchenType = ""  # string
LandSurface = 0  # int/float
HasLivingRoom = False  # bool
has_garden = False  # bool
garden_area = 0  # int/float
ParkingIndoor = 0  # int
ParkingOutdoor = 0  # int
HasAirConditioning = False  # bool
HasArmoredDoor = False  # bool
HasVisiophone = False  # bool
HasOffice = False  # bool
ToiletCount = 0  # int
has_swimming = False  # bool
HasFireplace = False  # bool
has_terrace = False  # bool
terrace_area = 0  # int/float
EpcScore = ""  # int/float ou None si inconnu

def markdown20(text):
    sl.markdown(f"<span style='font-size:20px'>{text}</span>", unsafe_allow_html=True)

sl.title("ImmoEliza")
sl.header("🏠 Real Estate Price Prediction 🏠")
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

BuildingCondition = sl.selectbox("Which condition?", p.df.buildingCondition.dropna().unique())
BuildingConstructionYear = sl.number_input(label="Construction year", min_value=1850, max_value=2050, value= 2000)
FacadeCount = sl.number_input(label= "Number of facades", min_value=1, max_value=20)

sl.markdown("----")
sl.subheader("📍 Geographical informations")

Province = sl.selectbox("Province", ['Brussels', 'Luxembourg', 'Antwerp', 'Flemish Brabant',
       'East Flanders', 'West Flanders', 'Liège', 'Walloon Brabant',
       'Limburg', 'Namur', 'Hainaut'])
Locality = sl.selectbox("Locality", sorted(p.df.locality.unique()))
Postcode = sl.selectbox("Postcode", sorted(p.df.postCode.unique()))

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

sl.markdown("----")
sl.subheader('📐 Surfaces')

markdown20('What\'s the size of your property?')
HabitableSurface = sl.slider("Habitable surface", 0, 800, 200, step=25)
markdown20('What\'s the land surface?')
LandSurface = sl.slider("Land surface", 0, 2000, 150, step=25)

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
    terrace_area = None

has_swimming = sl.checkbox('Has a swimmingpool')

flood_zone = sl.selectbox("There is a kind of flood zone?", p.df.floodZoneType.dropna().unique())

sl.markdown("----")
sl.subheader('♻️ Energy')
HasHeatPump = sl.checkbox('Has heat pump')
HasPhotovoltaicPanels = sl.checkbox('Has photovoltaic panels')
HasThermicPanels = sl.checkbox('Has thermic panels')
HasAirConditioning = sl.checkbox('Has air conditionning')
EpcScore = sl.selectbox('What\'s the EPC (PEB) score?', ["A++", "A+", "A", "B", "C", "D", "E", "F", "G"])
HeatingType = sl.selectbox('What\'s the heating type?', sorted(p.df.heatingType.dropna().unique()))


sl.markdown("----")
sl.subheader('✨ Extra')
HasFireplace = sl.checkbox('Has a fire place')
HasArmoredDoor = sl.checkbox('Has an armored door')
HasVisiophone = sl.checkbox('Has a visiophone')
ParkingIndoor = sl.slider(label= 'Parking places indoor', min_value=0, max_value=5)
ParkingOutdoor = sl.slider(label= 'Parking places outdoor', min_value=0, max_value=3)


sl.markdown("----")
if sl.button('Prediction'):
    input_dict = { 
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
    
    def convert_to_python(elem):
        if isinstance(elem, np.generic):
            return elem.item()
        return elem 
    
    input_dict = {key : convert_to_python(value) for key, value in input_dict.items()}

    
    response = requests.post(
        "http://localhost:8502",
        json=input_dict
    )

    if response.status_code == 200:
        sl.success(f"Résultat : {response.json()}")

    else: 
        sl.error(f"Erreur API : {response.text}")
