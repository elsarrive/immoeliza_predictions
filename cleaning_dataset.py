
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df = pd.read_csv('dataset/Kangaroo.csv')
df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

# # CLEANING FUNCTION

def cleaning_dataframe(df, df_giraffe = False, is_training = True):
    """
    This function is cleaning the dataframe. The steps are:
    #### Mapping of EPC score (A -> G) into the mean of EPC classification (kWh/m2.year) by region (Wallonia / Flanders / Brussels) & remove unwanted EPC
        df[epcScore] (object)  -> df[epc_enum] (float64)

    #### Removing ahberant values
        df['variable'] >= value (float64 / int64) -> np.nan
    
    #### Fill NaN in booleans columns and mapping true/false in boolean columns
        df['variable_bool'] (NaN) -> df['variable_bool'] = 'False' (object)
        df['variable_bool'] (object) -> df['variable_bool'] (int64)

    #### Summation of the parking counts
        df[ParkingCountIndoor] (float64) + df[ParkingCountOutdoor] (float64) -> df[ParkingCount] (float64) 
    
    #### Label-encoding for categories 
        df[subtype] = subtype (object) -> df[subtype_group] = group of subtypes (int64)
        df[province] (object) -> df[province_mapping] (int64)
        
        df[type] = type (object) -> df[isHouse] (float64)
        df[buildingCondition] (object) -> df[buildingCondition_mapping] (float64) 
        df[floodZoneType] (object) -> df[floodZoneType_mapping] (float64) 
        df[heatingType] (object) -> df[heatingType_mapping] (float64) 
        df[kitchenType] (object) -> df[kitchenType_mapping] (float64)
        
        df[facedeCount] (float64) -> df[facadeCount_mapping] (category)
        df[buildingConstructionYear] = years (float64) -> df[buildingConstructionYear_mapping] (category)

    #### Missing values for gardenSurface or terraceSurface
        0 if hasGarden or hasTerrace is 0

    #### Drop rows where there is no bathroomCount or bedroomCount 

    #### Merge 4 columns from data.csv (df_giraffe) where df_giraffe['propertyId'] == df['id']
        df_giraffe['latitude', 'longitude'] (float64)
        df_giraffe['primaryEnergyConsumptionPerSqm', 'cadastralIncome'] (int64)

    #### Filter the rows with a price margin (in a range of 50 000€ to 1 000 000€)

    #### Remove columns we don't use

    """
    #EPC SCORE
    epc_unwanted = ['C_A', 'F_C', 'G_C', 'D_C', 'F_D', 'E_C', 'G_E', 'E_D', 'C_B', 'X', 'G_F']
    df_epc = df[~df['epcScore'].isin(epc_unwanted)].copy()

    wallonia_provinces = ['Liège', 'Walloon Brabant', 'Namur', 'Hainaut', 'Luxembourg']
    flanders_provinces = ['Antwerp', 'Flemish Brabant', 'East Flanders', 'West Flanders', 'Limburg']

    wallonia_epc_map = {
        'A++' : 0,
        'A+' : 30,
        'A' : 65,
        'B' : 125,
        'C' : 200,
        'D' : 300,
        'E' : 375,
        'F' : 450,
        'G' : 510
    }
    flanders_epc_map = {
        'A++' : 0,
        'A+' : 0,
        'A' : 50,
        'B' : 150,
        'C' : 250,
        'D' : 350,
        'E' : 450,
        'F' : 500,
        'G' : 510,
    }
    brussels_epc_map = { 
        'A++' : 0,
        'A+' : 0,
        'A' : 45,
        'B' : 75,
        'C' : 125, 
        'D' : 175,
        'E' : 250,
        'F' : 300,
        'G' : 350,
    }

    df_epc.loc[df_epc['province'].isin(wallonia_provinces), 'epc_enum'] = df_epc['epcScore'].map(wallonia_epc_map).apply(pd.to_numeric)
    df_epc.loc[df_epc['province'].isin(flanders_provinces), 'epc_enum'] = df_epc['epcScore'].map(flanders_epc_map).apply(pd.to_numeric)
    df_epc.loc[df_epc['province'] == 'Brussels', 'epc_enum'] = df_epc['epcScore'].map(brussels_epc_map).apply(pd.to_numeric)

    # REMOVE ERROS / TOO BIG VALUES
    df_without_outliers = df_epc.copy()
    df_without_outliers.loc[df_without_outliers['bedroomCount'] >= 100, 'bedroomCount'] = np.nan
    df_without_outliers.loc[df_without_outliers['bathroomCount'] >= 100, 'bathroomCount'] = np.nan
    df_without_outliers.loc[df_without_outliers['toiletCount'] >= 25, 'toiletCount'] = np.nan
    df_without_outliers.loc[df_without_outliers['habitableSurface'] >= 600, 'habitableSurface'] = np.nan
    df_without_outliers.loc[df_without_outliers['landSurface'] >= 1000, 'landSurface'] = np.nan
    df_without_outliers.loc[df_without_outliers['gardenSurface'] >= 500, 'gardenSurface'] = np.nan
    df_without_outliers.loc[df_without_outliers['terraceSurface'] >= 250, 'terraceSurface'] = np.nan
    df_without_outliers.loc[df_without_outliers['parkingCountIndoor'] >= 10, 'parkingCountIndoor'] = 1
    df_without_outliers.loc[df_without_outliers['parkingCountOutdoor'] >= 10, 'parkingCountOutdoor'] = 1
    
    # BOOLEANS COLUMNS (FILL NaN + MAPPING)
    booleans_columns = ['hasAttic', 'hasBasement', 'hasDressingRoom', 'hasDiningRoom', 'hasLift', 'hasHeatPump', 'hasPhotovoltaicPanels', 'hasThermicPanels', 'hasLivingRoom', 'hasGarden', 'hasAirConditioning', 'hasArmoredDoor', 'hasVisiophone', 'hasOffice', 'hasSwimmingPool', 'hasFireplace', 'hasTerrace']
    df_without_outliers.loc[:, booleans_columns] = df_without_outliers[booleans_columns].fillna('False')

    boolean_to_num = {'True' : 1, 
        'true' : 1, 
        'False' : 0, 
        'false' : 0,
        False : 0,
        True: 1}

    for col in booleans_columns:
        df_without_outliers.loc[:, col] = df_without_outliers[col].replace('nan', 'false') #parfois je pense qu'il est écrit nan et c'est pas NaN
        df_without_outliers.loc[:, col] = df_without_outliers[col].map(boolean_to_num)

    
    # CREATE A PARKING COLUMN (INDOOR + OUTDOOR) 
    df_with_park = df_without_outliers.copy()
    df_with_park.loc[:, 'parkingCount'] = df_with_park[['parkingCountIndoor', 'parkingCountOutdoor']].sum(axis=1, min_count=1)

    # LABEL-ENCODING FOR CATEGORIES
    ## subgroup
    subtype_to_group = {
    "APARTMENT": 1,
    "FLAT_STUDIO": 1,
    "DUPLEX": 1,
    "TRIPLEX": 1,
    "PENTHOUSE": 1,
    "LOFT": 1,
    "SERVICE_FLAT": 1,
    "GROUND_FLOOR": 1,
    "KOT": 1,
    "MIXED_USE_BUILDING": 1,

    "HOUSE": 2,
    "TOWN_HOUSE": 2,
    "VILLA": 2,
    "CHALET": 2,
    "BUNGALOW": 2,
    "COUNTRY_COTTAGE": 2,

    "MANOR_HOUSE": 3,
    "MANSION": 3,
    "EXCEPTIONAL_PROPERTY": 3,
    "CASTLE": 3,
    "FARMHOUSE": 3,

    "APARTMENT_BLOCK": 4,
    "APARTMENT_GROUP" : 4,
    "HOUSE_GROUP": 4,

    "OTHER_PROPERTY": 5,
    "PAVILION": 5
    }

    df_subtype = df_with_park.copy()
    df_subtype.loc[:, 'subtype_group'] = df_subtype['subtype'].map(subtype_to_group).apply(pd.to_numeric)

    ## building construction year
    df_year = df_subtype.copy()
    years_bins = [1850, 1875, 1900, 1925, 1950, 1975, 2000, 2025, 2050]
    years_labels = [1, 2, 3, 4, 5, 6, 7, 8]
    df_year.loc[:, 'buildingConstructionYear_mapping'] = pd.cut(
    df_year['buildingConstructionYear'], 
    bins= years_bins,
    labels= years_labels)

    ## type
    df_type = df_year.copy()
    df_type.loc[:, 'isHouse'] = df_type['type'].map({ 
        "APARTMENT" : 0,
        "HOUSE" : 1
    }).apply(pd.to_numeric)

    ## provinces
    df_province = df_type.copy()
    df_province.province.unique()
    province_mapping = { 
        'Brussels' : 1,
        'Luxembourg' : 2,
        'Antwerp' : 3,
        'Flemish Brabant' : 4,
        'East Flanders' : 5,
        'West Flanders' : 6,
        'Liège' : 7,
        'Walloon Brabant' : 8,
        'Limburg' : 9,
        'Namur' : 10,
        'Hainaut' : 11
    }

    df_province.loc[:, 'province_mapping'] = df_province['province'].map(province_mapping).apply(pd.to_numeric)

    ## building condition
    df_condition = df_province.copy()
    condition_mapping = { 
            'GOOD' : 5,
            'TO_BE_DONE_UP' : 4,
            'AS_NEW' : 3,
            'JUST_RENOVATED' : 2,
            'TO_RENOVATE' : 1,
            'TO_RESTORE' : 0
        }

    df_condition.loc[:, 'buildingCondition_mapping'] = df_condition['buildingCondition'].map(condition_mapping).apply(pd.to_numeric)

    ## flood zone type
    df_flood = df_condition.copy()
    floodZoneType_mapping = {
            "NON_FLOOD_ZONE": 1,
            "POSSIBLE_N_CIRCUMSCRIBED_WATERSIDE_ZONE": 2,
            "CIRCUMSCRIBED_WATERSIDE_ZONE": 3,
            "POSSIBLE_N_CIRCUMSCRIBED_FLOOD_ZONE": 4,
            "POSSIBLE_FLOOD_ZONE": 5,
            "CIRCUMSCRIBED_FLOOD_ZONE": 6,
            "RECOGNIZED_FLOOD_ZONE": 7,
            "RECOGNIZED_N_CIRCUMSCRIBED_WATERSIDE_FLOOD_ZONE": 8,
            "RECOGNIZED_N_CIRCUMSCRIBED_FLOOD_ZONE": 9
            }

    df_flood.loc[:, 'floodZoneType_mapping'] = df_flood['floodZoneType'].map(floodZoneType_mapping).apply(pd.to_numeric)

    ## heating type
    df_heat = df_flood.copy()
    heatingType_mapping = { 
        'GAS' : 1, 
        'FUELOIL' : 2, 
        'ELECTRIC' : 3, 
        'PELLET' : 4, 
        'WOOD' : 4, 
        'SOLAR' : 4, 
        'CARBON' : 4
    }

    df_heat.loc[:, 'heatingType_mapping'] = df_heat['heatingType'].map(heatingType_mapping).apply(pd.to_numeric)

    ## kitchen type
    df_kitchen = df_heat.copy()
    kitchenType_mapping = {
        "NOT_INSTALLED": 0,
        "USA_UNINSTALLED": 0,

        "USA_SEMI_EQUIPPED": 1,
        "SEMI_EQUIPPED": 1,

        "USA_INSTALLED": 2,
        "INSTALLED": 2,

        "USA_HYPER_EQUIPPED": 3,
        "HYPER_EQUIPPED": 3,
        }

    df_kitchen.loc[:, 'kitchenType_mapping'] = df_kitchen.kitchenType.map(kitchenType_mapping).apply(pd.to_numeric)

    ## facade count
    df_facade = df_kitchen.copy()
    facedeCount_bins = [0, 1, 2, 3, 4, float('inf')]
    facedeCount_labels = [1, 2, 3, 4, 5]

    df_facade.loc[:, 'facadecount_mapping'] = pd.cut( 
        df_facade['facedeCount'],
        bins = facedeCount_bins,
        labels = facedeCount_labels,
        include_lowest= True
    ).apply(pd.to_numeric)

    # MISSING VALUE FOR ...SURFACE
    df_hasG_hasT = df_facade.copy()
    df_hasG_hasT.loc[df_hasG_hasT['hasGarden']  == 0, 'gardenSurface'] = 0
    df_hasG_hasT.loc[df_hasG_hasT['hasTerrace']  == 0, 'terraceSurface'] = 0

    # IF NO BATHROOM OR BEDROOM COUNT: DROP
    df_no_bed_bath = df_hasG_hasT.copy()
    if is_training: 
        df_no_bed_bath = df_no_bed_bath.dropna(subset=['bedroomCount', 'bathroomCount'])

    else:
        df_no_bed_bath = df_no_bed_bath
    
    # IMPORT FROM AN OTHER DATASET
    df_with_giraffe = df_no_bed_bath.copy()

    giraffe_cols = [
    "latitude",
    "longitude",
    "primaryEnergyConsumptionPerSqm",
    "cadastralIncome"
    ]

    if is_training : 
        df_giraffe = pd.read_csv('data.csv')
        df_with_giraffe = df_with_giraffe.merge(
        df_giraffe[['propertyId'] + giraffe_cols],  
        how='inner',
        left_on='id',
        right_on='propertyId'
        )

    else: 
        df_giraffe = pd.read_csv('dataset/data.csv')

        for col in giraffe_cols:
            if col not in df_with_giraffe.columns:
                if col in ['latitude', 'longitude']:
                    df_with_giraffe[col] = df_giraffe[col].median()

                else: 
                    df_with_giraffe[col] = 0

    # PRICE MARGIN
    df_margin = df_with_giraffe.copy()
    if is_training: 
        df_margin = df_margin[(df_margin['price'] >= 50000) & (df_margin['price'] <= 1000000)]
    
    else: 
        df_margin = df_margin

    # REMOVE ROWS
    df_dropped = df_margin.drop(columns=['url', 'type', 'subtype', 'province', 'monthlyCost', 'diningRoomSurface', 'buildingCondition', 'buildingConstructionYear', 'facedeCount', 'floorCount', 'streetFacadeWidth', 'floodZoneType', 'kitchenType', 'hasBalcony', 'gardenOrientation', 'terraceOrientation', 'accessibleDisabledPeople', 'epcScore', 'kitchenSurface', 'livingRoomSurface', 'roomCount', 'parkingCountIndoor', 'parkingCountOutdoor', 'locality', 'propertyId', 'hasTerrace', 'hasGarden', 'heatingType'], errors='ignore')

    return df_dropped


def stats(X_train):
    """
    Stats dictionary (stats_from_X_train) stores all the information needed to impute missing values.
    
    # Fill_with_mode:
    For each of these columns, it calculates the most frequent value (mode) in X_train.
    
    # Fill_with_median : 
    For quantitative columns such as gardenSurface and terraceSurface, the function calculates the median on X_train.

    
    # Regression imputation for important features: 
    - toiletCount imputed via the variables bedroomCount, bathroomCount, habitableSurface
    - habitableSurface imputed via the variables bathroomCount, bedroomCount, parkingCount, isHouse
    - landSurface imputed via the variables habitableSurface, gardenSurface, parkingCount

    Each imputation is made exclusively on X_train.
    
    # The function returns stats_from_X_train, containing :
    - Calculated modes
    - The medians
    - The three regressive imputation models already trained
        
    """

    stats_from_X_train = {}
    stats_from_X_train['imputers'] = {}

    # fill with mode 
    fill_with_mode = [
        'heatingType_mapping', 'facadecount_mapping', 'floodZoneType_mapping',
        'buildingCondition_mapping', 'buildingConstructionYear_mapping', 'epc_enum', 'kitchenType_mapping'
    ]
    
    stats_from_X_train['mode'] = {
        col: X_train[col].mode()[0] for col in fill_with_mode
        }

    # fill with median 
    fill_with_median = ['gardenSurface', 'terraceSurface']
    stats_from_X_train['median'] = {
        col: X_train[col].median() for col in fill_with_median
        }

    # regression imputation 
    stats_from_X_train['imputers']['toiletCount'] = IterativeImputer(max_iter=10, random_state=0, initial_strategy='median').fit(
        X_train[['bedroomCount', 'bathroomCount', 'toiletCount', 'habitableSurface']]
    )
    stats_from_X_train['imputers']['habitableSurface'] = IterativeImputer(max_iter=10, random_state=0, initial_strategy='median').fit(
        X_train[['bathroomCount', 'bedroomCount', 'parkingCount', 'isHouse', 'habitableSurface']]
    )
    stats_from_X_train['imputers']['landSurface'] = IterativeImputer(max_iter=10, random_state=0, initial_strategy='median').fit(
        X_train[['habitableSurface', 'gardenSurface', 'parkingCount', 'landSurface']]
    )

    return stats_from_X_train

def transform_cleaning_traintestsplit(df, stats, is_training=True):
    """
    This function is for imputation of missing values.
    It has to be made on the dataframe splitted into x_test & x_train to don't have any data leakage.

    #### Regression imputation for important features 
    NaN in toiletCount, habitableSurface, parkingCount, landSurface -> mode (float64)

    #### Imputation with the mode for missing values in label variables 
    Note: facadecount_mapping & buildingConstructionYear_mapping (categrory) -> (int64)

    #### Imputation with the median for missing values in continuous variables

    #### Remove rows where there is any missing 
    """

    # REGRESSION IMPUTATION
    for feature, features_related in [('toiletCount', ['bedroomCount', 'bathroomCount', 'toiletCount', 'habitableSurface']),
                      ('habitableSurface', ['bathroomCount', 'bedroomCount', 'parkingCount', 'isHouse', 'habitableSurface']),
                      ('landSurface', ['habitableSurface', 'gardenSurface', 'parkingCount', 'landSurface'])]:
        
        if all(col in df.columns for col in features_related):
            df_imputed = stats['imputers'][feature].transform(df[features_related])
            df_imputed = pd.DataFrame(df_imputed, columns=features_related).round()
            for col in features_related:
                df[col] = df_imputed[col].values

    # MODE IMPUTATION 
    for col, mode_val in stats['mode'].items():
        if col in df.columns:
            df[col] = df[col].fillna(mode_val)

    # MEDIAN IMPUTATION 
    for col, median_val in stats['median'].items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)
   
    # CATEGORY TO INT
    for col in ['facadecount_mapping', 'buildingConstructionYear_mapping']:
        if col in df.columns:
            df[col] = df[col].astype(int, errors='ignore')

    # REMOVE ROWS WHERE NAN
    if is_training :
        df_final = df[~df.isna().any(axis=1)]

    else: 
        df_final = df

    return df_final

