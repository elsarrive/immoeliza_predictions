import streamlit as sl
import predictions_elsa as p

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
EpcScore = sl.selectbox('What\'s the EPC (PEB) score?', p.df['epcScore'][~p.df['epcScore'].isin(p.epc_unwanted)].dropna().sort_values().unique())
HeatingType = sl.selectbox('What\'s the heating type?', sorted(p.df.heatingType.dropna().unique()))


sl.markdown("----")
sl.subheader('✨ Extra')
HasFireplace = sl.checkbox('Has a fire place')
HasArmoredDoor = sl.checkbox('Has an armored door')
HasVisiophone = sl.checkbox('Has a visiophone')
ParkingIndoor = sl.slider(label= 'Parking places indoor', min_value=0, max_value=5)
ParkingOutdoor = sl.slider(label= 'Parking places outdoor', min_value=0, max_value=3)



